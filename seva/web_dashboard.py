"""
Web Dashboard可视化模块（阶段37）
对应第一性原理：易用 = 可视化操作，无需命令行

提供：
- WebDashboard: Web仪表盘（HTTP服务器+HTML界面）
- start_dashboard(): 一键启动Web仪表盘
- generate_html(): 生成静态HTML报告

功能:
- 实时查看Agent状态
- 运行控制（启动/停止/重置）
- 指标可视化
- 配置查看
- 诊断报告

使用示例:
    from seva import start_dashboard
    start_dashboard(agent, port=8080)  # 启动后访问 http://localhost:8080
"""
import os
import json
import time
import threading
import html
from typing import Any, Dict, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from .api import SEVAFactory, run, get_status
from .diagnostics import Diagnoser, diagnose_agent
from .monitoring import Dashboard, MetricsCollector


class DashboardHandler(BaseHTTPRequestHandler):
    """Web Dashboard HTTP请求处理器"""

    # 类级别共享状态
    agent = None
    dashboard = None
    collector = None
    running_stats = {"steps_run": 0, "last_run_time": 0}

    def log_message(self, format, *args):
        """禁用默认日志输出"""
        pass

    def do_GET(self):
        """处理GET请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/status":
            self._serve_json(self._get_status())
        elif path == "/api/metrics":
            self._serve_json(self._get_metrics())
        elif path == "/api/config":
            self._serve_json(self._get_config())
        elif path == "/api/diagnose":
            self._serve_json(self._get_diagnosis())
        elif path == "/api/scenarios":
            self._serve_json(self._get_scenarios())
        elif path == "/api/run":
            steps = int(params.get("steps", ["10"])[0])
            self._serve_json(self._run_agent(steps))
        elif path == "/api/reset":
            self._serve_json(self._reset_agent())
        else:
            self._serve_404()

    def _serve_html(self):
        """提供HTML页面"""
        html_content = self._generate_html()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html_content.encode("utf-8"))

    def _serve_json(self, data: Dict):
        """提供JSON响应"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False,
                                    default=str).encode("utf-8"))

    def _serve_404(self):
        """404响应"""
        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("404 Not Found".encode("utf-8"))

    def _get_status(self) -> Dict:
        """获取Agent状态"""
        if not self.agent:
            return {"error": "Agent未创建"}
        return get_status(self.agent)

    def _get_metrics(self) -> Dict:
        """获取指标"""
        if not self.agent:
            return {"error": "Agent未创建"}
        if self.dashboard:
            self.dashboard.collect_from_agent()
        return self.agent.get_metrics()

    def _get_config(self) -> Dict:
        """获取配置"""
        if not self.agent:
            return {"error": "Agent未创建"}
        return self.agent.config.to_dict()

    def _get_diagnosis(self) -> Dict:
        """获取诊断报告"""
        if not self.agent:
            return {"error": "Agent未创建"}
        diagnoser = Diagnoser(self.agent)
        return diagnoser.diagnose()

    def _get_scenarios(self) -> Dict:
        """获取场景列表"""
        from .scenarios import list_scenarios_detailed
        return {"scenarios": list_scenarios_detailed()}

    def _run_agent(self, steps: int) -> Dict:
        """运行Agent"""
        if not self.agent:
            return {"error": "Agent未创建"}
        start_time = time.time()
        stats = run(self.agent, steps=steps, verbose=False)
        elapsed = time.time() - start_time
        self.running_stats["steps_run"] += steps
        self.running_stats["last_run_time"] = elapsed
        return {
            "steps_completed": stats["steps_completed"],
            "total_reward": stats["total_real_reward"],
            "avg_cycle_time": stats["avg_cycle_time"],
            "elapsed_time": elapsed,
        }

    def _reset_agent(self) -> Dict:
        """重置Agent"""
        if not self.agent:
            return {"error": "Agent未创建"}
        self.agent.reset()
        self.dashboard = Dashboard(self.agent)
        return {"success": True, "message": "Agent已重置"}

    def _generate_html(self) -> str:
        """生成HTML页面"""
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SEVA Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
               Roboto, sans-serif; background: #f5f5f5; color: #333; }
        .header { background: linear-gradient(135deg, #667eea, #764ba2);
                  color: white; padding: 20px; text-align: center; }
        .header h1 { font-size: 24px; margin-bottom: 5px; }
        .header p { font-size: 14px; opacity: 0.9; }
        .container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
        .card { background: white; border-radius: 8px; padding: 20px;
                margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h2 { font-size: 18px; margin-bottom: 15px; color: #667eea; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit,
                minmax(250px, 1fr)); gap: 15px; }
        .metric { background: #f8f9fa; padding: 15px; border-radius: 6px;
                  border-left: 4px solid #667eea; }
        .metric-label { font-size: 12px; color: #666; margin-bottom: 5px; }
        .metric-value { font-size: 20px; font-weight: bold; color: #333; }
        .controls { display: flex; gap: 10px; flex-wrap: wrap; }
        .btn { padding: 10px 20px; border: none; border-radius: 6px;
               cursor: pointer; font-size: 14px; transition: 0.3s; }
        .btn-primary { background: #667eea; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-warning { background: #ffc107; color: #333; }
        .btn:hover { opacity: 0.9; transform: translateY(-1px); }
        .status-badge { display: inline-block; padding: 4px 12px;
                       border-radius: 12px; font-size: 12px; }
        .status-healthy { background: #d4edda; color: #155724; }
        .status-error { background: #f8d7da; color: #721c24; }
        #result { margin-top: 10px; padding: 10px; background: #f8f9fa;
                  border-radius: 4px; display: none; }
        .loading { text-align: center; padding: 20px; color: #999; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left;
                 border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SEVA Dashboard</h1>
        <p>自驱永续进化预测修正Agent系统</p>
    </div>
    <div class="container">
        <div class="card">
            <h2>控制面板</h2>
            <div class="controls">
                <button class="btn btn-primary" onclick="runAgent(10)">运行10步</button>
                <button class="btn btn-primary" onclick="runAgent(50)">运行50步</button>
                <button class="btn btn-primary" onclick="runAgent(100)">运行100步</button>
                <button class="btn btn-warning" onclick="resetAgent()">重置</button>
                <button class="btn btn-success" onclick="refreshAll()">刷新</button>
            </div>
            <div id="result"></div>
        </div>

        <div class="card">
            <h2>Agent状态</h2>
            <div class="grid" id="status-grid">
                <div class="loading">加载中...</div>
            </div>
        </div>

        <div class="card">
            <h2>运行指标</h2>
            <div class="grid" id="metrics-grid">
                <div class="loading">加载中...</div>
            </div>
        </div>

        <div class="card">
            <h2>诊断报告</h2>
            <div id="diagnosis">
                <div class="loading">加载中...</div>
            </div>
        </div>
    </div>

    <script>
        async function fetchJSON(url) {
            const response = await fetch(url);
            return response.json();
        }

        async function refreshStatus() {
            const data = await fetchJSON('/api/status');
            const grid = document.getElementById('status-grid');
            if (data.error) {
                grid.innerHTML = '<p>' + data.error + '</p>';
                return;
            }
            let html = '';
            for (const [key, value] of Object.entries(data)) {
                const displayValue = typeof value === 'boolean' ?
                    (value ? '是' : '否') : value;
                html += '<div class="metric"><div class="metric-label">' +
                        key + '</div><div class="metric-value">' +
                        displayValue + '</div></div>';
            }
            grid.innerHTML = html;
        }

        async function refreshMetrics() {
            const data = await fetchJSON('/api/metrics');
            const grid = document.getElementById('metrics-grid');
            if (data.error) {
                grid.innerHTML = '<p>' + data.error + '</p>';
                return;
            }
            const keys = ['cycle_count', 'total_reward',
                         'avg_prediction_error', 'avg_cycle_time', 'mode'];
            let html = '';
            for (const key of keys) {
                if (key in data) {
                    let val = data[key];
                    if (key === 'avg_cycle_time') {
                        val = (val * 1000).toFixed(2) + 'ms';
                    } else if (typeof val === 'number') {
                        val = val.toFixed(4);
                    }
                    html += '<div class="metric"><div class="metric-label">' +
                            key + '</div><div class="metric-value">' +
                            val + '</div></div>';
                }
            }
            grid.innerHTML = html;
        }

        async function refreshDiagnosis() {
            const data = await fetchJSON('/api/diagnose');
            const div = document.getElementById('diagnosis');
            if (data.error) {
                div.innerHTML = '<p>' + data.error + '</p>';
                return;
            }
            const badgeClass = data.healthy ? 'status-healthy' : 'status-error';
            const badgeText = data.healthy ? '健康' : '异常';
            let html = '<p>健康状态: <span class="status-badge ' + badgeClass +
                       '">' + badgeText + '</span></p>';
            html += '<p>问题数: ' + data.issues_count + '</p>';
            html += '<p>警告数: ' + data.warnings_count + '</p>';
            if (data.issues && data.issues.length > 0) {
                html += '<h3 style="margin-top:10px">问题列表:</h3><ul>';
                data.issues.forEach(function(issue) {
                    html += '<li>' + issue + '</li>';
                });
                html += '</ul>';
            }
            div.innerHTML = html;
        }

        async function runAgent(steps) {
            const result = document.getElementById('result');
            result.style.display = 'block';
            result.innerHTML = '运行中...';
            const data = await fetchJSON('/api/run?steps=' + steps);
            if (data.error) {
                result.innerHTML = '错误: ' + data.error;
            } else {
                result.innerHTML = '完成: ' + data.steps_completed +
                    '步, 奖励: ' + data.total_reward.toFixed(4) +
                    ', 耗时: ' + data.elapsed_time.toFixed(2) + 's';
            }
            refreshAll();
        }

        async function resetAgent() {
            await fetchJSON('/api/reset');
            refreshAll();
            document.getElementById('result').style.display = 'none';
        }

        function refreshAll() {
            refreshStatus();
            refreshMetrics();
            refreshDiagnosis();
        }

        refreshAll();
        setInterval(refreshAll, 5000);
    </script>
</body>
</html>"""


class WebDashboard:
    """
    Web Dashboard服务器

    使用示例:
        dashboard = WebDashboard(agent, port=8080)
        dashboard.start()  # 启动服务器
        # 访问 http://localhost:8080
        dashboard.stop()   # 停止服务器
    """

    def __init__(self, agent=None, host: str = "localhost",
                 port: int = 8080, preset: str = "easy"):
        self.agent = agent
        self.host = host
        self.port = port
        self.preset = preset
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

        # 如果没有提供agent，创建一个
        if self.agent is None:
            self.agent = SEVAFactory.create(preset)

        # 设置handler的类级别状态
        DashboardHandler.agent = self.agent
        DashboardHandler.dashboard = Dashboard(self.agent)
        DashboardHandler.collector = MetricsCollector()

    def start(self, blocking: bool = False):
        """启动Web服务器"""
        self.server = HTTPServer((self.host, self.port), DashboardHandler)

        if blocking:
            print(f"SEVA Web Dashboard已启动: http://{self.host}:{self.port}")
            print("按Ctrl+C停止")
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self.thread = threading.Thread(
                target=self.server.serve_forever, daemon=True
            )
            self.thread.start()
            print(f"SEVA Web Dashboard已启动: http://{self.host}:{self.port}")

    def stop(self):
        """停止Web服务器"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None
        print("Web Dashboard已停止")

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.server is not None

    def get_url(self) -> str:
        """获取访问URL"""
        return f"http://{self.host}:{self.port}"


def start_dashboard(agent=None, host: str = "localhost",
                    port: int = 8080, blocking: bool = True,
                    preset: str = "easy") -> WebDashboard:
    """
    一键启动Web Dashboard

    参数:
        agent: Agent实例（None则自动创建）
        host: 监听地址
        port: 监听端口
        blocking: 是否阻塞（True则前台运行）
        preset: 自动创建Agent时使用的预设

    返回:
        WebDashboard实例

    示例:
        from seva import start_dashboard
        start_dashboard(port=8080)  # 访问 http://localhost:8080
    """
    dashboard = WebDashboard(agent=agent, host=host, port=port, preset=preset)
    dashboard.start(blocking=blocking)
    return dashboard


def generate_html_report(agent, output_file: Optional[str] = None) -> str:
    """
    生成静态HTML报告（不需要服务器）

    参数:
        agent: Agent实例
        output_file: 输出文件路径（None则返回字符串）

    返回:
        HTML内容字符串（如果output_file为None）

    示例:
        from seva import generate_html_report
        html = generate_html_report(agent, "report.html")
    """
    status = get_status(agent)
    metrics = agent.get_metrics()
    diagnoser = Diagnoser(agent)
    diagnosis = diagnoser.diagnose()

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>SEVA报告 - {time.strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; background: #f5f5f5; }}
        .card {{ background: white; padding: 20px; margin: 20px 0;
                border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #667eea; }}
        h2 {{ color: #667eea; margin-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left;
                 border-bottom: 1px solid #eee; }}
        .badge {{ padding: 4px 12px; border-radius: 12px;
                 font-size: 12px; }}
        .healthy {{ background: #d4edda; color: #155724; }}
        .error {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <h1>SEVA Agent报告</h1>
    <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="card">
        <h2>状态摘要</h2>
        <table>
            <tr><th>指标</th><th>值</th></tr>
            <tr><td>周期数</td><td>{status.get('cycle_count', 0)}</td></tr>
            <tr><td>运行模式</td><td>{status.get('mode', 'N/A')}</td></tr>
            <tr><td>健康状态</td>
                <td><span class="badge {'healthy' if status.get('healthy') else 'error'}">
                    {'健康' if status.get('healthy') else '异常'}</span></td></tr>
            <tr><td>总奖励</td><td>{status.get('total_reward', 0)}</td></tr>
            <tr><td>平均周期时间</td><td>{status.get('avg_cycle_time_ms', 0)}ms</td></tr>
            <tr><td>任务完成</td><td>{status.get('tasks_completed', 0)}</td></tr>
            <tr><td>技能总数</td><td>{status.get('skills_total', 0)}</td></tr>
            <tr><td>进化评估</td><td>{status.get('evolution_evaluations', 0)}</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>诊断报告</h2>
        <p>健康: <span class="badge {'healthy' if diagnosis.get('healthy') else 'error'}">
            {'是' if diagnosis.get('healthy') else '否'}</span></p>
        <p>问题数: {diagnosis.get('issues_count', 0)}</p>
        <p>警告数: {diagnosis.get('warnings_count', 0)}</p>
    </div>

    <div class="card">
        <h2>详细指标</h2>
        <table>
            <tr><th>指标</th><th>值</th></tr>
            <tr><td>平均预测误差</td>
                <td>{metrics.get('avg_prediction_error', 0):.4f}</td></tr>
            <tr><td>平均惊奇值</td>
                <td>{metrics.get('avg_surprise', 0):.4f}</td></tr>
            <tr><td>动作分布</td>
                <td>{metrics.get('action_distribution', {})}</td></tr>
        </table>
    </div>
</body>
</html>"""

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        return html_content
    return html_content
