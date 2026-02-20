import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from app.logging_config import configure_logging
from app.routes import router

# Relatório de drift gerado em monitoring/drift_report.html (raiz do projeto).
DRIFT_REPORT_PATH = Path(__file__).resolve().parent.parent / "monitoring" / "drift_report.html"

def create_app() -> FastAPI:
    # Configura logging estruturado em JSON para toda a aplicação.
    configure_logging()
    logging.getLogger(__name__).info("Aplicação FastAPI inicializada", extra={"component": "startup"})

    app = FastAPI(title="Abandono API", version="1.0.0")
    app.include_router(router, prefix="/api")
    return app


app = create_app()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring_dashboard():
    """
    Sirve o dashboard de drift gerado por monitoring/drift_report.py.
    """
    if not DRIFT_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Relatório de drift não encontrado. Rode monitoring/drift_report.py para gerá-lo.",
        )
    return FileResponse(DRIFT_REPORT_PATH, media_type="text/html")
