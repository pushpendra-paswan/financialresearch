import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
PAGES = ["index", "companies", "company", "watchlists", "alerts", "notifications", "team"]
COMMON_EXPORTS = {
    "api",
    "saveToken",
    "hasToken",
    "requireLogin",
    "renderNav",
    "renderPager",
    "el",
    "showMessage",
}

FOOTER_NOTICE = "Data from SEC EDGAR and Yahoo Finance. For research only, not investment advice."

html_files = sorted(FRONTEND_DIR.glob("*.html"))
js_files = sorted(FRONTEND_DIR.glob("*.js"))


# ---------- Static serving ----------


def test_frontend_files_exist() -> None:
    # Guards against the other tests passing on an empty folder
    expected = {"style.css", "common.js"}
    expected |= {f"{page}.html" for page in PAGES} | {f"{page}.js" for page in PAGES}
    found = {path.name for path in FRONTEND_DIR.iterdir()}
    assert found == expected


def test_app_root_serves_index(client: TestClient) -> None:
    response = client.get("/app/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


@pytest.mark.parametrize("page", PAGES)
def test_pages_are_served_without_a_token(client: TestClient, page: str) -> None:
    response = client.get(f"/app/{page}.html")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_stylesheet_content_type(client: TestClient) -> None:
    response = client.get("/app/style.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


@pytest.mark.parametrize("name", ["common", *PAGES])
def test_scripts_are_javascript(client: TestClient, name: str) -> None:
    response = client.get(f"/app/{name}.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_root_redirects_to_app(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/app/"


def test_unknown_static_file_is_json_404(client: TestClient) -> None:
    response = client.get("/app/nope.html")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_path_traversal_is_rejected(client: TestClient) -> None:
    response = client.get("/app/%2e%2e/app/config.py")
    assert 400 <= response.status_code < 500
    assert "Settings" not in response.text


def test_mount_does_not_shadow_api_routes(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200


# ---------- Static guards on the frontend files ----------

FORBIDDEN_JS_PATTERNS = [
    r"innerHTML",
    r"outerHTML",
    r"insertAdjacentHTML",
    r"document\.write",
    r"\beval\(",
    r"new Function",
]


@pytest.mark.parametrize("path", js_files, ids=lambda path: path.name)
def test_js_has_no_dangerous_apis(path: Path) -> None:
    text = path.read_text()
    for pattern in FORBIDDEN_JS_PATTERNS:
        assert not re.search(pattern, text), f"{path.name} contains forbidden pattern {pattern}"


@pytest.mark.parametrize("path", js_files, ids=lambda path: path.name)
def test_network_and_storage_only_in_common_js(path: Path) -> None:
    if path.name == "common.js":
        return
    text = path.read_text()
    for pattern in [r"\bfetch\(", r"localStorage", r"sessionStorage"]:
        assert not re.search(pattern, text), f"{path.name} contains {pattern}; use common.js"


def test_common_js_exports_exactly_the_eight_names() -> None:
    text = (FRONTEND_DIR / "common.js").read_text()
    export_lines = re.findall(r"^export\b.*$", text, flags=re.MULTILINE)
    names = re.findall(r"^export\s+(?:async\s+)?function\s+(\w+)", text, flags=re.MULTILINE)
    assert len(export_lines) == len(names), f"common.js has a non-function export: {export_lines}"
    assert set(names) == COMMON_EXPORTS, f"common.js exports {sorted(names)}"
    assert len(names) == len(set(names))


@pytest.mark.parametrize("path", js_files, ids=lambda path: path.name)
def test_js_imports_only_common_js(path: Path) -> None:
    text = path.read_text()
    assert not re.search(r"\bimport\s*\(", text), f"{path.name} uses a dynamic import"
    import_statements = re.findall(r"^import\b[^;]*;", text, flags=re.MULTILINE)
    import_lines = re.findall(r"^import\b", text, flags=re.MULTILINE)
    assert len(import_statements) == len(import_lines), f"{path.name} has an unusual import"
    for statement in import_statements:
        assert re.search(r"""from\s+["']\./common\.js["']\s*;$""", statement), (
            f"{path.name} imports from somewhere other than ./common.js: {statement}"
        )


@pytest.mark.parametrize("path", html_files, ids=lambda path: path.name)
def test_html_security_rules(path: Path) -> None:
    text = path.read_text()

    # The CSP meta tag is the first element in <head> and keeps everything on our own origin
    assert re.search(
        r"<head>\s*<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self'; "
        r"base-uri 'self'; form-action 'self'\">",
        text,
    ), f"{path.name}: the CSP meta tag is missing or not the first element in <head>"

    tags = re.findall(r"<[a-zA-Z][^>]*>", text)
    for tag in tags:
        assert not re.search(r"\son[a-zA-Z]+\s*=", tag), f"{path.name}: event handler in {tag}"
        assert not re.search(r"\sstyle\s*=", tag), f"{path.name}: style attribute in {tag}"

    assert "<style" not in text, f"{path.name}: inline <style> element"

    scripts = re.findall(r"<script\b[^>]*>.*?</script>", text, flags=re.DOTALL)
    assert len(scripts) == 1, f"{path.name}: expected exactly one script tag"
    for script in scripts:
        assert re.match(r"<script\b[^>]*>\s*</script>$", script), f"{path.name}: inline script"
        assert 'type="module"' in script, f"{path.name}: script is not type=module"
        source = re.search(r'\ssrc="([^"]*)"', script)
        assert source, f"{path.name}: script without src"
        assert (FRONTEND_DIR / source.group(1)).is_file(), f"{path.name}: missing {source.group(1)}"

    # No absolute or protocol-relative addresses on script and link tags
    for tag in re.findall(r"<(?:script|link)\b[^>]*>", text):
        for address in re.findall(r'\s(?:src|href)="([^"]*)"', tag):
            assert not re.match(r"(https?:)?//", address), f"{path.name}: external address in {tag}"
            assert not address.startswith("/"), f"{path.name}: absolute path in {tag}"


@pytest.mark.parametrize("path", html_files, ids=lambda path: path.name)
def test_html_has_standard_parts(path: Path) -> None:
    text = path.read_text()
    assert "<title>" in text, f"{path.name}: no title"
    assert '<header id="nav"></header>' in text, f"{path.name}: no nav header"
    assert 'id="message"' in text, f"{path.name}: no message area"
    assert FOOTER_NOTICE in text, f"{path.name}: footer notice is missing"


def test_css_has_no_imports_or_external_urls() -> None:
    text = (FRONTEND_DIR / "style.css").read_text()
    assert "@import" not in text, "style.css contains @import"
    for address in re.findall(r"url\(\s*['\"]?([^)'\"]+)", text):
        assert not re.match(r"(https?:)?//", address), f"style.css has an external url({address})"
