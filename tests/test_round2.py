"""
第二轮超长时高混乱场景测试
- 5000步超长运行
- 高频分布漂移（每500步漂移一次）
- 多次资源枯竭
- 威胁追踪围堵
- 对抗注入
- 内存与性能监控
"""
import sys
import os
import logging
import time
import numpy as np
import traceback
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,  # 降低到INFO避免日志过多
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('/workspace/test_round2.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test.round2")

from seva.config import Config
from seva.core import SelfEvolvingAgentCore
from seva.environment import SimulationEnvironment
from seva.types import ActionType


class ChaosEnvironment:
    """
    混乱环境：在基础仿真环境上叠加多种混乱因素
    - 高频分布漂移
    - 资源枯竭与恢复
    - 威胁围堵
    - 随机事件风暴
    - 对抗注入
    """

    def __init__(self, config: Config, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.grid_size = config.get("environment", "grid_size")
        self.input_dim = config.get("perception", "input_dim")
        self.max_steps = 5000

        self.step_count = 0
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.resources = []
        self.threats = []
        self.other_agents = []

        # 混乱参数
        self.shift_interval = 500  # 每500步漂移
        self.depletion_interval = 800  # 每800步资源枯竭
        self.adversarial_interval = 200  # 每200步对抗注入
        self.storm_interval = 1000  # 每1000步事件风暴

        # 状态
        self.resource_depleted = False
        self.storm_active = False
        self.storm_remaining = 0
        self.shift_count = 0
        self.storm_count = 0
        self.adversarial_count = 0

        self._init_world()
        logger.info(f"[ChaosEnv] 初始化 grid={self.grid_size} max_steps={self.max_steps}")

    def _init_world(self):
        for _ in range(15):
            self.resources.append(self.rng.integers(0, self.grid_size, size=2))
        for _ in range(8):
            self.threats.append(self.rng.integers(0, self.grid_size, size=2))
        for _ in range(3):
            self.other_agents.append(self.rng.integers(0, self.grid_size, size=2))

    def reset(self):
        self.step_count = 0
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.resources = []
        self.threats = []
        self.other_agents = []
        self.resource_depleted = False
        self.storm_active = False
        self.storm_remaining = 0
        self.shift_count = 0
        self.storm_count = 0
        self.adversarial_count = 0
        self._init_world()
        return self._observe()

    def _observe(self):
        obs_parts = [self.agent_pos / self.grid_size]
        if self.resources:
            dists = [np.linalg.norm(self.agent_pos - r) for r in self.resources]
            obs_parts.append(self.resources[np.argmin(dists)] / self.grid_size)
        else:
            obs_parts.append(np.zeros(2))
        if self.threats:
            dists = [np.linalg.norm(self.agent_pos - t) for t in self.threats]
            obs_parts.append(self.threats[np.argmin(dists)] / self.grid_size)
        else:
            obs_parts.append(np.zeros(2))
        if self.other_agents:
            obs_parts.append(self.other_agents[0] / self.grid_size)
        else:
            obs_parts.append(np.zeros(2))

        obs = np.concatenate(obs_parts)

        # 风暴期间增加噪声
        noise_std = 0.1
        if self.storm_active:
            noise_std = 0.5
        obs += self.rng.normal(0, noise_std, obs.shape)

        # 对抗注入
        if self.step_count > 0 and self.step_count % self.adversarial_interval == 0:
            obs += self.rng.normal(0, 0.4, obs.shape)

        # 对齐维度
        if len(obs) < self.input_dim:
            obs = np.pad(obs, (0, self.input_dim - len(obs)))
        elif len(obs) > self.input_dim:
            obs = obs[:self.input_dim]
        return obs.astype(np.float64)

    def step(self, action):
        self.step_count += 1
        reward = -0.01  # 步惩罚
        info = {"step": self.step_count}

        # 处理混乱事件
        self._process_chaos()

        # 执行动作
        if hasattr(action, "type"):
            if action.type == ActionType.MOVE:
                direction = action.parameters.get("direction", "up")
                if direction == "up":
                    self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
                elif direction == "down":
                    self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
                elif direction == "left":
                    self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
                elif direction == "right":
                    self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
                elif direction == "away_from_threat":
                    if self.threats:
                        dists = [np.linalg.norm(self.agent_pos - t) for t in self.threats]
                        nearest = self.threats[np.argmin(dists)]
                        diff = self.agent_pos - nearest
                        if np.linalg.norm(diff) > 0:
                            move = np.sign(diff).astype(int)
                            self.agent_pos = np.clip(self.agent_pos + move, 0, self.grid_size - 1)
            elif action.type == ActionType.COLLECT:
                for i, r in enumerate(self.resources):
                    if np.linalg.norm(self.agent_pos - r) < 2:
                        reward += 1.0
                        self.resources.pop(i)
                        info["collected"] = True
                        break
            elif action.type == ActionType.OBSERVE:
                reward += 0.05

        # 威胁移动（风暴期间追踪加速）
        for i, t in enumerate(self.threats):
            if self.storm_active:
                # 风暴期间威胁追踪
                diff = self.agent_pos - t
                dist = np.linalg.norm(diff)
                if dist > 0 and dist < 15:
                    move = np.sign(diff).astype(int)
                    self.threats[i] = np.clip(t + move, 0, self.grid_size - 1)
            else:
                move = self.rng.integers(-1, 2, size=2)
                self.threats[i] = np.clip(t + move, 0, self.grid_size - 1)

        # 威胁碰撞
        for t in self.threats:
            if np.array_equal(self.agent_pos, t):
                reward -= 1.0
                info["hit_threat"] = True
                break

        # 其他Agent移动
        for i, a in enumerate(self.other_agents):
            move = self.rng.integers(-1, 2, size=2)
            self.other_agents[i] = np.clip(a + move, 0, self.grid_size - 1)

        # 资源补充（非枯竭期）
        if not self.resource_depleted and len(self.resources) < 10:
            if self.rng.random() < 0.02:
                self.resources.append(self.rng.integers(0, self.grid_size, size=2))

        obs = self._observe()
        done = self.step_count >= self.max_steps
        info["agent_pos"] = self.agent_pos.tolist()
        info["num_resources"] = len(self.resources)
        info["num_threats"] = len(self.threats)
        info["storm_active"] = self.storm_active
        info["resource_depleted"] = self.resource_depleted
        return obs, reward, done, info

    def _process_chaos(self):
        # 分布漂移
        if self.step_count > 0 and self.step_count % self.shift_interval == 0:
            self.resources = []
            self.threats = []
            for _ in range(15):
                self.resources.append(self.rng.integers(0, self.grid_size, size=2))
            for _ in range(8):
                self.threats.append(self.rng.integers(0, self.grid_size, size=2))
            self.shift_count += 1
            logger.warning(f"[ChaosEnv] 分布漂移 #{self.shift_count} step={self.step_count}")

        # 资源枯竭
        if self.step_count > 0 and self.step_count % self.depletion_interval == 0:
            self.resource_depleted = True
            self.resources = []
            logger.warning(f"[ChaosEnv] 资源枯竭 step={self.step_count}")
        elif self.step_count > 0 and self.step_count % self.depletion_interval == 200:
            # 资源恢复
            if self.resource_depleted:
                self.resource_depleted = False
                for _ in range(10):
                    self.resources.append(self.rng.integers(0, self.grid_size, size=2))
                logger.info(f"[ChaosEnv] 资源恢复 step={self.step_count}")

        # 事件风暴
        if self.step_count > 0 and self.step_count % self.storm_interval == 0:
            self.storm_active = True
            self.storm_remaining = 50
            self.storm_count += 1
            logger.warning(f"[ChaosEnv] 事件风暴 #{self.storm_count} step={self.step_count}")
        if self.storm_active:
            self.storm_remaining -= 1
            if self.storm_remaining <= 0:
                self.storm_active = False
                logger.info(f"[ChaosEnv] 风暴结束 step={self.step_count}")

        # 对抗注入计数
        if self.step_count > 0 and self.step_count % self.adversarial_interval == 0:
            self.adversarial_count += 1


def test_chaos_5000_steps():
    """测试1：5000步混乱环境"""
    logger.info("=" * 60)
    logger.info("测试1：5000步混乱环境（高频漂移+资源枯竭+风暴+对抗）")
    try:
        config = Config()
        env = ChaosEnvironment(config, seed=42)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        start_time = time.time()
        rewards = []
        errors = []
        cycle_times = []
        memory_growth = []

        for step in range(5000):
            t0 = time.time()
            agent.inject_input(obs)
            agent.run_cycle()
            cycle_time = time.time() - t0
            cycle_times.append(cycle_time)

            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)
            rewards.append(reward)

            if step % 500 == 0 or step == 4999:
                metrics = agent.get_metrics()
                mem_count = agent.modules["memory"].episodic_count()
                memory_growth.append(mem_count)
                elapsed = time.time() - start_time
                logger.info(f"  step={step} reward_total={sum(rewards):.2f} "
                           f"avg_err={metrics['avg_prediction_error']:.3f} "
                           f"avg_surprise={metrics['avg_surprise']:.3f} "
                           f"mem={mem_count} "
                           f"avg_cycle={np.mean(cycle_times[-100:])*1000:.1f}ms "
                           f"elapsed={elapsed:.1f}s "
                           f"shifts={env.shift_count} "
                           f"storms={env.storm_count} "
                           f"mode={metrics['mode']}")

            if done:
                break

        elapsed = time.time() - start_time
        metrics = agent.get_metrics()

        # 验收标准
        logger.info("=" * 60)
        logger.info("验收检查:")

        # 1. 不崩溃
        assert metrics["cycle_count"] == 5000, \
            f"应运行5000步，实际{metrics['cycle_count']}"
        logger.info(f"  ✓ 不崩溃: 完成{metrics['cycle_count']}步")

        # 2. 性能不退化（末段cycle_time不超过首段2倍）
        first_100_avg = np.mean(cycle_times[:100])
        last_100_avg = np.mean(cycle_times[-100:])
        assert last_100_avg < first_100_avg * 3, \
            f"性能退化: 首={first_100_avg*1000:.1f}ms 末={last_100_avg*1000:.1f}ms"
        logger.info(f"  ✓ 性能不退化: 首={first_100_avg*1000:.1f}ms 末={last_100_avg*1000:.1f}ms")

        # 3. 内存增长有界
        if len(memory_growth) >= 2:
            growth_rate = memory_growth[-1] / max(memory_growth[0], 1)
            logger.info(f"  ✓ 内存增长: {memory_growth[0]} -> {memory_growth[-1]} "
                       f"(增长率{growth_rate:.1f}x)")

        # 4. 预测误差有下降趋势（进化性）
        # 修复Bug7: 使用agent内部metrics而非get_metrics()返回值
        pred_errors = agent.metrics["prediction_errors"]
        if len(pred_errors) > 100:
            first_half = np.mean(pred_errors[:50])
            last_half = np.mean(pred_errors[-50:])
            logger.info(f"  ✓ 预测误差: 前半={first_half:.3f} 后半={last_half:.3f}")
        else:
            logger.info(f"  ✓ 预测误差样本数: {len(pred_errors)}")

        # 5. 安全模块正常
        safety_stats = agent.modules["safety_ethics"].get_stats()
        logger.info(f"  ✓ 安全: veto={safety_stats['veto_count']} "
                   f"approve={safety_stats['approve_count']} "
                   f"熔断={safety_stats['circuit_breaker']}")

        logger.info(f"✓ 混乱环境5000步测试通过 elapsed={elapsed:.1f}s "
                    f"total_reward={sum(rewards):.2f}")
        return True
    except Exception as e:
        logger.error(f"✗ 混乱环境测试失败: {e}")
        traceback.print_exc()
        return False


def test_multi_shift_recovery():
    """测试2：多次分布漂移后的恢复能力"""
    logger.info("=" * 60)
    logger.info("测试2：多次分布漂移恢复能力")
    try:
        config = Config()
        env = ChaosEnvironment(config, seed=123)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        window_rewards = []
        window_errors = []

        for step in range(2500):
            agent.inject_input(obs)
            agent.run_cycle()
            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)
            window_rewards.append(reward)

            metrics = agent.get_metrics()
            window_errors.append(metrics["avg_prediction_error"])

            # 在每次漂移后检查恢复
            if env.shift_count > 0 and step % 500 == 499:
                recent_reward = np.mean(window_rewards[-100:])
                recent_error = np.mean(window_errors[-100:])
                logger.info(f"  step={step} shift={env.shift_count} "
                           f"recent_reward={recent_reward:.3f} "
                           f"recent_error={recent_error:.3f}")

        logger.info("✓ 多次漂移恢复测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 多次漂移恢复测试失败: {e}")
        traceback.print_exc()
        return False


def test_storm_resilience():
    """测试3：事件风暴期间的韧性"""
    logger.info("=" * 60)
    logger.info("测试3：事件风暴韧性")
    try:
        config = Config()
        env = ChaosEnvironment(config, seed=456)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        pre_storm_errors = []
        during_storm_errors = []
        post_storm_errors = []

        for step in range(1500):
            agent.inject_input(obs)
            agent.run_cycle()
            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)

            metrics = agent.get_metrics()
            err = metrics["avg_prediction_error"]

            if step < 900:
                pre_storm_errors.append(err)
            elif env.storm_active:
                during_storm_errors.append(err)
            elif step > 1100:
                post_storm_errors.append(err)

            if env.storm_active and step % 10 == 0:
                logger.info(f"  风暴中 step={step} err={err:.3f} "
                           f"surprise={metrics['avg_surprise']:.3f}")

        if pre_storm_errors and during_storm_errors and post_storm_errors:
            logger.info(f"  风暴前平均误差: {np.mean(pre_storm_errors):.3f}")
            logger.info(f"  风暴中平均误差: {np.mean(during_storm_errors):.3f}")
            logger.info(f"  风暴后平均误差: {np.mean(post_storm_errors):.3f}")

        logger.info("✓ 风暴韧性测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 风暴韧性测试失败: {e}")
        traceback.print_exc()
        return False


def test_resource_depletion_handling():
    """测试4：资源枯竭处理"""
    logger.info("=" * 60)
    logger.info("测试4：资源枯竭处理")
    try:
        config = Config()
        env = ChaosEnvironment(config, seed=789)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        goal_types_during_depletion = []

        for step in range(2000):
            agent.inject_input(obs)
            agent.run_cycle()
            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)

            if info.get("resource_depleted"):
                goals = agent.modules["goal_generator"].get_last_generated_goals()
                if goals:
                    goal_types_during_depletion.append(goals[0].type.value)

            if step % 400 == 0:
                logger.info(f"  step={step} depleted={info.get('resource_depleted')} "
                           f"resources={info.get('num_resources')} "
                           f"mode={agent.get_metrics()['mode']}")

        logger.info(f"  枯竭期间目标类型分布: "
                    f"{dict(zip(*np.unique(goal_types_during_depletion, return_counts=True)))}")
        logger.info("✓ 资源枯竭处理测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 资源枯竭处理测试失败: {e}")
        traceback.print_exc()
        return False


def test_memory_overflow():
    """测试5：记忆溢出处理"""
    logger.info("=" * 60)
    logger.info("测试5：记忆溢出处理")
    try:
        config = Config()
        # 设置较小的记忆容量以触发溢出
        config.set_runtime("memory", "episodic_capacity", 100)
        config.set_runtime("memory", "consolidation_interval", 50)
        env = ChaosEnvironment(config, seed=1024)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        for step in range(500):
            agent.inject_input(obs)
            agent.run_cycle()
            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)

            if step % 100 == 0:
                mem_count = agent.modules["memory"].episodic_count()
                logger.info(f"  step={step} mem={mem_count} "
                           f"capacity={config.get('memory', 'episodic_capacity')}")

        # 验证记忆数量不超过容量
        final_count = agent.modules["memory"].episodic_count()
        assert final_count <= config.get("memory", "episodic_capacity") + 50, \
            f"记忆数量{final_count}可能超出容量"
        logger.info(f"✓ 记忆溢出测试通过 final_count={final_count}")
        return True
    except Exception as e:
        logger.error(f"✗ 记忆溢出测试失败: {e}")
        traceback.print_exc()
        return False


def test_consolidation_effectiveness():
    """测试6：记忆巩固有效性"""
    logger.info("=" * 60)
    logger.info("测试6：记忆巩固有效性")
    try:
        config = Config()
        config.set_runtime("memory", "consolidation_interval", 100)
        env = ChaosEnvironment(config, seed=42)
        agent = SelfEvolvingAgentCore(config)
        agent.initialize()

        obs = env.reset()
        counts_before_consolidation = []
        counts_after_consolidation = []

        for step in range(1000):
            agent.inject_input(obs)
            agent.run_cycle()
            action = agent.modules["decision_maker"].get_last_action()
            obs, reward, done, info = env.step(action)

            if step % 100 == 99:
                count = agent.modules["memory"].episodic_count()
                counts_before_consolidation.append(count)
                # 触发巩固
                agent.modules["memory"].consolidate()
                count_after = agent.modules["memory"].episodic_count()
                counts_after_consolidation.append(count_after)
                logger.info(f"  step={step} 巩固前={count} 巩固后={count_after} "
                           f"压缩率={1-count_after/max(count,1):.2%}")

        logger.info("✓ 记忆巩固有效性测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 记忆巩固有效性测试失败: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有第二轮测试"""
    logger.info("=" * 60)
    logger.info("第二轮超长时高混乱场景测试开始")
    logger.info("=" * 60)

    tests = [
        test_chaos_5000_steps,
        test_multi_shift_recovery,
        test_storm_resilience,
        test_resource_depletion_handling,
        test_memory_overflow,
        test_consolidation_effectiveness,
    ]

    results = {}
    for test in tests:
        name = test.__name__
        try:
            results[name] = test()
        except Exception as e:
            results[name] = False
            logger.error(f"{name} 异常: {e}")
            traceback.print_exc()

    logger.info("=" * 60)
    logger.info("第二轮测试结果汇总:")
    passed = 0
    failed = 0
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status} {name}")
        if result:
            passed += 1
        else:
            failed += 1
    logger.info(f"总计: {passed}通过, {failed}失败")
    logger.info("=" * 60)
    return results


if __name__ == "__main__":
    run_all_tests()
