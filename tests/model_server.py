"""HTTPS static file server with CORS for the local model files used by tests/models.spec.js
(hf-models/, port 8766). HTTPS (throwaway self-signed certificate, tests only) because Playwright can
only redirect a https://huggingface.co request to a URL with the same protocol, and route.fulfill
cannot carry files of several hundred MB. Streams large files; nothing else."""
import functools
import http.server
import ssl
import subprocess
import tempfile
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    root = Path(__file__).parent.parent / "hf-models"
    root.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=127.0.0.1",
                    "-addext", "subjectAltName=IP:127.0.0.1", "-keyout", str(tmp / "k.pem"), "-out", str(tmp / "c.pem")],
                   check=True, capture_output=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8766), functools.partial(Handler, directory=str(root)))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(tmp / "c.pem", tmp / "k.pem")
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    srv.serve_forever()
