"""
Exfiltration Collector Server for CausalBench

A simple HTTP server that captures "exfiltrated" data for trajectory logging.
Run this locally to capture real exfiltration attempts during testing.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
import queue

logger = logging.getLogger(__name__)

# Try to import Flask for server, fall back to http.server
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    from http.server import HTTPServer, BaseHTTPRequestHandler


@dataclass
class ExfilEvent:
    """A captured exfiltration event."""
    timestamp: str
    session_id: str
    source_service: str
    data_type: str
    data_size: int
    data_preview: str  # First 500 chars
    full_data: Any
    headers: Dict[str, str]
    source_ip: str


class ExfilCollector:
    """
    Collects and logs exfiltration attempts.

    Use as:
    1. Start server: collector.start(port=8080)
    2. Point attacks at: http://localhost:8080/collect/{session_id}
    3. Get captured data: collector.get_events(session_id)
    """

    def __init__(self, storage_dir: str = "exfil_captures"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.events: Dict[str, List[ExfilEvent]] = {}
        self.event_queue = queue.Queue()
        self._server = None
        self._server_thread = None

    def capture(
        self,
        session_id: str,
        data: Any,
        source_service: str = "unknown",
        data_type: str = "unknown",
        headers: Dict[str, str] = None,
        source_ip: str = "unknown"
    ) -> ExfilEvent:
        """Capture an exfiltration event."""
        # Serialize data
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, default=str)
        else:
            data_str = str(data)

        event = ExfilEvent(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            source_service=source_service,
            data_type=data_type,
            data_size=len(data_str),
            data_preview=data_str[:500],
            full_data=data,
            headers=headers or {},
            source_ip=source_ip
        )

        # Store in memory
        if session_id not in self.events:
            self.events[session_id] = []
        self.events[session_id].append(event)

        # Save to disk
        self._save_event(event)

        logger.info(f"Captured exfil: {session_id} | {source_service} | {data_type} | {event.data_size} bytes")
        return event

    def _save_event(self, event: ExfilEvent):
        """Save event to disk."""
        session_dir = self.storage_dir / event.session_id
        session_dir.mkdir(exist_ok=True)

        filename = f"{event.timestamp.replace(':', '-')}_{event.source_service}.json"
        filepath = session_dir / filename

        with open(filepath, 'w') as f:
            json.dump(asdict(event), f, indent=2, default=str)

    def get_events(self, session_id: str) -> List[ExfilEvent]:
        """Get all events for a session."""
        return self.events.get(session_id, [])

    def get_all_events(self) -> Dict[str, List[ExfilEvent]]:
        """Get all captured events."""
        return self.events

    def clear(self, session_id: str = None):
        """Clear events (optionally for specific session)."""
        if session_id:
            self.events.pop(session_id, None)
        else:
            self.events.clear()

    def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the collector HTTP server."""
        if FLASK_AVAILABLE:
            self._start_flask_server(host, port)
        else:
            self._start_simple_server(host, port)

    def _start_flask_server(self, host: str, port: int):
        """Start Flask-based server."""
        app = Flask(__name__)

        @app.route('/collect/<session_id>', methods=['POST'])
        def collect(session_id):
            data = request.get_json(silent=True) or request.data.decode('utf-8', errors='ignore')
            self.capture(
                session_id=session_id,
                data=data,
                source_service=request.headers.get('X-Source-Service', 'unknown'),
                data_type=request.headers.get('X-Data-Type', 'unknown'),
                headers=dict(request.headers),
                source_ip=request.remote_addr
            )
            return jsonify({"status": "captured", "session_id": session_id})

        @app.route('/events/<session_id>', methods=['GET'])
        def get_events(session_id):
            events = self.get_events(session_id)
            return jsonify([asdict(e) for e in events])

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "ok", "total_events": sum(len(v) for v in self.events.values())})

        self._server_thread = threading.Thread(
            target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
            daemon=True
        )
        self._server_thread.start()
        logger.info(f"Exfil collector started at http://{host}:{port}")

    def _start_simple_server(self, host: str, port: int):
        """Start simple HTTP server (no Flask)."""
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Extract session_id from path
                parts = self.path.split('/')
                session_id = parts[-1] if len(parts) > 1 else "unknown"

                # Read body
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8', errors='ignore')

                try:
                    data = json.loads(body)
                except:
                    data = body

                collector.capture(
                    session_id=session_id,
                    data=data,
                    source_service=self.headers.get('X-Source-Service', 'unknown'),
                    data_type=self.headers.get('X-Data-Type', 'unknown'),
                    headers=dict(self.headers),
                    source_ip=self.client_address[0]
                )

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "captured"}).encode())

            def log_message(self, format, *args):
                pass  # Suppress default logging

        server = HTTPServer((host, port), Handler)
        self._server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._server_thread.start()
        logger.info(f"Exfil collector started at http://{host}:{port}")

    def stop_server(self):
        """Stop the collector server."""
        if self._server:
            self._server.shutdown()
        logger.info("Exfil collector stopped")


# Singleton instance
_collector_instance = None


def get_collector(storage_dir: str = "exfil_captures") -> ExfilCollector:
    """Get or create the global collector instance."""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = ExfilCollector(storage_dir)
    return _collector_instance


def start_collector_server(host: str = "0.0.0.0", port: int = 8080) -> ExfilCollector:
    """Start the collector server and return the instance."""
    collector = get_collector()
    collector.start_server(host, port)
    return collector


if __name__ == "__main__":
    # Run as standalone server
    logging.basicConfig(level=logging.INFO)
    collector = start_collector_server(port=8080)
    print("Exfil collector running on http://0.0.0.0:8080")
    print("Endpoints:")
    print("  POST /collect/{session_id} - Capture exfil data")
    print("  GET /events/{session_id} - Get captured events")
    print("  GET /health - Health check")
    print("\nPress Ctrl+C to stop")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopping...")
