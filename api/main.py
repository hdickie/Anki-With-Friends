from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi import Path

import os
import time
import sqlite3
from typing import Optional

from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates



import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from fastapi import Path as FPath, Request

from datetime import datetime

import logging
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

logger.info("🚀 logger.info uvicorn logger test")

logging.getLogger("uvicorn.error").info("UVICORN ERROR LOGGER TEST")
logging.getLogger("uvicorn").info("UVICORN LOGGER TEST")
print('print TEST')


# Where the analysis JSON lives (persist this with a Docker volume)
ANALYSIS_DIR = Path(os.getenv("ANALYSIS_DIR", "/data"))
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = ANALYSIS_DIR / "zipf_vocab_analysis.json"

# The script you want to run when button is pressed
# (adjust this path to your repo layout)
SCRIPT_PATH = Path(os.getenv("VOCAB_SCRIPT_PATH", "/app/scripts/run_vocab_analysis.py"))

ANALYSIS_DIR = Path("/data")
OUTPUT_JSON = ANALYSIS_DIR / "zipf_vocab_analysis.json"


def _file_info(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": st.st_size,
        "modified_epoch": st.st_mtime,
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
    }


def _load_json_pretty(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False)



from pathlib import Path
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent  # .../app/api
TEMPLATES_DIR = BASE_DIR / "templates"      # .../app/api/templates

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI()
from api.routes_analysis import router as analysis_router
from fastapi import APIRouter
router = APIRouter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(analysis_router)

DB_PATH = os.getenv("DB_PATH", "/data/app.db")


def get_conn() -> sqlite3.Connection:
    # check_same_thread=False is fine for small demos; for real apps,
    # use proper connection handling / a thread-safe approach.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL
            )
            """
        )
        con.commit()


@app.on_event("startup")
def on_startup():
    init_db()


class NoteIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": True, "db_path": DB_PATH}


@app.post("/notes")
def create_note(note: NoteIn):
    with get_conn() as con:
        cur = con.execute("INSERT INTO notes(text) VALUES (?)", (note.text,))
        con.commit()
        return {"id": cur.lastrowid, "text": note.text}


@app.get("/notes")
def list_notes(limit: int = 50, q: Optional[str] = None):
    limit = max(1, min(limit, 200))
    with get_conn() as con:
        if q:
            rows = con.execute(
                "SELECT id, text FROM notes WHERE text LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{q}%", limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT id, text FROM notes ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {"notes": [dict(r) for r in rows]}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": "Hume",   # later: from auth
            "year": 2026,
        }
    )

@app.get("/debug-template", response_class=HTMLResponse)
def debug_template(request: Request):
    return templates.TemplateResponse(
        "base.html",
        {"request": request, "user": "Hume", "year": 2026},
    )

# ✅ Whitelist: only these pages can be rendered
ALLOWED_PAGES = {
    "index",
    "decks",
    "friends",
    "about",
    "debug-template",
}

def _as_simple_dict(items: Any) -> Dict[str, Any]:
    """
    Convert Starlette QueryParams / FormData to a plain dict.
    If a key appears multiple times, keep the last.
    """
    out: Dict[str, Any] = {}
    for k, v in items.multi_items():
        out[k] = v
    return out

def render_page(
    request: Request,
    page: str,
    *,
    user: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> HTMLResponse:
    # 1) validate page
    if page not in ALLOWED_PAGES:
        raise HTTPException(status_code=404, detail="Page not found")

    # 2) map page -> template filename
    template_name = f"{page}.html"

    # 3) base context
    context: Dict[str, Any] = {
        "request": request,
        "app_name": "Anki with Friends",
        "user": user or "Hume",  # later: from auth/session
        "year": dt.datetime.now().year,
    }

    # 4) merge extra data (query/form/etc.)
    if extra:
        context.update(extra)

    return templates.TemplateResponse(template_name, context)

@app.get("/ui/{page_path:path}", response_class=HTMLResponse)
async def ui_get(
    request: Request,
    page_path: str = FPath(...),
):
    # page_path can include slashes; must sanitize
    if ".." in page_path or page_path.startswith("/"):
        raise HTTPException(status_code=404)

    if page_path.endswith(".html"):
        page_path = page_path[:-5]

    template_name = f"ui/{page_path}.html"
    return templates.TemplateResponse(
        template_name,
        {"request": request, "user": "Hume", "year": 2026},
    )


@app.post("/ui/{page}", response_class=HTMLResponse)
async def ui_post(page: str, request: Request):
    # collect form fields from <form method="post">...</form>
    form = await request.form()
    form_data = _as_simple_dict(form)

    # do something with form_data here (write to DB, call API, etc.)
    # then render the template with "submitted" data available
    return render_page(
        request,
        page,
        extra={
            "form": form_data,
            "submitted": True,
        },
    )


@app.get("/debug/hostdata")
def debug_hostdata():
    base = "/hostdata"

    if not os.path.exists(base):
        raise HTTPException(status_code=500, detail="hostdata not mounted")

    out = []

    for root, dirs, files in os.walk(base):
        for name in files:
            path = os.path.join(root, name)
            stat = os.stat(path)
            out.append({
                "path": path,
                "size_bytes": stat.st_size,
                "modified": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(stat.st_mtime),
                ),
            })

    return {
        "cwd": os.getcwd(),
        "files": out,
    }



# @router.get("/analysis/vocab", response_class=HTMLResponse)
# def vocab_page(request: Request):
#     logger.warning("GET vocab_page: OUTPUT_JSON=%s exists=%s", OUTPUT_JSON, OUTPUT_JSON.exists())
#     if OUTPUT_JSON.exists():
#         logger.warning("GET vocab_page: stat size=%s mtime=%s", OUTPUT_JSON.stat().st_size, OUTPUT_JSON.stat().st_mtime)

#     file_info = get_file_info(OUTPUT_JSON)
#     logger.warning("GET vocab_page: file_info=%s", file_info)

#     return templates.TemplateResponse(
#         "analysis/vocab.html",
#         {"request": request, "file_info": file_info},
#     )



# @router.post("/analysis/vocab/run")
# def vocab_run():
#     """
#     Runs the analysis script. Script should write OUTPUT_JSON.
#     Then redirect back to the UI page to display updated results.
#     """

#     # Basic “dev safety”: ensure script exists
#     if not SCRIPT_PATH.exists():
#         # Redirect with a simple marker; you can do proper flash messages later
#         return RedirectResponse(url="/analysis/vocab?error=missing_script", status_code=303)

#     # Run it. If your script writes to OUTPUT_JSON, great.
#     # If the script needs output path, pass it via env.
#     env = os.environ.copy()
#     env["ANALYSIS_DIR"] = str(ANALYSIS_DIR)
#     env["OUTPUT_JSON"] = str(OUTPUT_JSON)

#     logger.info("Vocab page file_info: %s", env["OUTPUT_JSON"])

#     try:
#         result = subprocess.run(
#             ["python", str(SCRIPT_PATH)],
#             cwd="/app",
#             env=env,
#             capture_output=True,
#             text=True,
#             timeout=300,   # 5 minutes; adjust as needed
#             check=False,
#         )
#     except subprocess.TimeoutExpired:
#         return RedirectResponse(url="/analysis/vocab?error=timeout", status_code=303)

#     # If script failed, redirect with error and stash a small log file for inspection.
#     if result.returncode != 0:
#         log_path = ANALYSIS_DIR / "zipf_vocab_analysis_last_error.log"
#         log_path.write_text(
#             f"returncode={result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
#             encoding="utf-8",
#         )
#         return RedirectResponse(url="/analysis/vocab?error=script_failed", status_code=303)

#     # Ensure output exists
#     if not OUTPUT_JSON.exists():
#         return RedirectResponse(url="/analysis/vocab?error=no_output", status_code=303)

#     return RedirectResponse(url="/analysis/vocab?ok=1", status_code=303)