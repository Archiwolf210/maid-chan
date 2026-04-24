"""
check_server.py — Digital Human v4
Watchdog / health checker для llama-server и python backend.
Используется из start.bat для ожидания готовности сервисов.

Usage:
  python check_server.py llama <port> [max_wait_sec]
  python check_server.py backend <port> [max_wait_sec]
"""
import json
import socket
import sys
import time
import urllib.error
import urllib.request


def _tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Быстрая проверка — слушает ли кто-то на порту."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except OSError:
        return False


def _http_get(url: str, timeout: float = 3.0):
    """Простой HTTP GET, возвращает (status_code, body_dict|None)."""
    try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = json.loads(resp.read().decode())
        return resp.status, body
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return None, None


def wait_for_llama_server(port: int, max_wait: int = 300) -> bool:
    """
    Ждёт полной загрузки llama-server:
    1. TCP порт открыт
    2. /health возвращает 200 и модель загружена
    """
    check_interval = 3
    elapsed = 0
    print(f"[*] Ожидание llama-server на порту {port}...")

    while elapsed < max_wait:
        if _tcp_open("127.0.0.1", port):
            status, body = _http_get(f"http://127.0.0.1:{port}/health")
            if status == 200 and body:
                # llama-server сигнализирует готовность через status или slots
                if body.get("status") == "ok" or "slots" in body:
                    print(f"[OK] llama-server готов! ({elapsed}с)")
                    return True
                # Если model_loaded_time есть — тоже ОК
                if body.get("model_loaded_time") is not None:
                    print(f"[OK] llama-server готов! ({elapsed}с)")
                    return True

        elapsed += check_interval
        if elapsed % 30 == 0:
            print(f"[.] Ещё ждём загрузку модели... ({elapsed}с)")
        time.sleep(check_interval)

    print(f"[ERROR] llama-server не запустился за {max_wait}с")
    return False


def wait_for_backend(port: int, max_wait: int = 60) -> bool:
    """
    Ждёт готовности Python backend:
    1. TCP порт открыт
    2. /health возвращает 200
    """
    check_interval = 1
    elapsed = 0
    print(f"[*] Ожидание backend на порту {port}...")

    while elapsed < max_wait:
        if _tcp_open("127.0.0.1", port):
            status, body = _http_get(f"http://127.0.0.1:{port}/health")
            if status == 200:
                print(f"[OK] Backend готов! ({elapsed}с)")
                return True

        elapsed += check_interval
        if elapsed % 10 == 0:
            print(f"[.] Ещё ждём backend... ({elapsed}с)")
        time.sleep(check_interval)

    print(f"[WARNING] Backend не ответил за {max_wait}с — проверь logs/error.log")
    return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python check_server.py <llama|backend> <port> [max_wait]")
        sys.exit(1)

    server_type = sys.argv[1].lower()
    port        = int(sys.argv[2])
    max_wait    = int(sys.argv[3]) if len(sys.argv) > 3 else (300 if server_type == "llama" else 60)

    if server_type == "llama":
        sys.exit(0 if wait_for_llama_server(port, max_wait) else 1)
    elif server_type == "backend":
        sys.exit(0 if wait_for_backend(port, max_wait) else 1)
    else:
        print(f"[ERROR] Неизвестный тип: {server_type}")
        sys.exit(1)
