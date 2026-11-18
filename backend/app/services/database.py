"""SQLite metadata persistence for HoloGen reconstruction projects."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("services.database")


class Base(DeclarativeBase):
    pass


class ProjectRecord(Base):
    """SQLite table storing reconstruction project metadata, status, and JSON report."""

    __tablename__ = "projects"

    project_id = Column(String(128), primary_key=True, index=True)
    name = Column(String(256), nullable=False)
    mode = Column(String(32), nullable=False, default="single_view")
    status = Column(String(32), nullable=False, default="uploaded")
    progress_pct = Column(Integer, nullable=False, default=0)
    stage_message = Column(String(512), nullable=False, default="Uploaded")
    source_image_path = Column(String(1024), nullable=False)
    output_dir = Column(String(1024), nullable=False)
    parameters_json = Column(Text, nullable=False, default="{}")
    report_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(String(64), nullable=False)
    updated_at = Column(String(64), nullable=False)


class DatabaseManager:
    """Manage SQLite connection and CRUD operations for projects."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.db_path = db_path or (settings.outputs_dir / "hologen.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False
        )
        self._seed_existing_baseline_runs(settings.outputs_dir)

    def _seed_existing_baseline_runs(self, outputs_dir: Path) -> None:
        """Import any existing CLI baseline runs in outputs/ into SQLite so they appear in UI."""
        if not outputs_dir.exists():
            return
        with self.SessionLocal() as session:
            for sub in sorted(outputs_dir.iterdir()):
                if not sub.is_dir():
                    continue
                rep_file = sub / "run_report.json"
                glb_file = sub / "scene.glb"
                if rep_file.exists() and glb_file.exists():
                    existing = session.get(ProjectRecord, sub.name)
                    if existing is None:
                        try:
                            rep_data = json.loads(rep_file.read_text(encoding="utf-8"))
                            ts = rep_data.get(
                                "timestamp_utc", datetime.now(timezone.utc).isoformat()
                            )
                            src_img = str((sub / "input_preprocessed.png").resolve())
                            rec = ProjectRecord(
                                project_id=sub.name,
                                name=sub.name,
                                mode="single_view",
                                status="completed",
                                progress_pct=100,
                                stage_message="Reconstruction complete",
                                source_image_path=src_img,
                                output_dir=str(sub.resolve()),
                                parameters_json=json.dumps(
                                    {
                                        "model_name": rep_data.get("depth_estimation", {}).get(
                                            "model_name", "depth_anything_v2_small"
                                        ),
                                        "quality_preset": rep_data.get("mesh", {}).get(
                                            "quality_preset", "viewer"
                                        ),
                                    }
                                ),
                                report_json=json.dumps(rep_data),
                                error_message=None,
                                created_at=ts,
                                updated_at=ts,
                            )
                            session.add(rec)
                        except Exception as exc:
                            logger.warning("Skipping seed of %s: %s", sub.name, exc)
            session.commit()

    def create_project(
        self,
        project_id: str,
        name: str,
        source_image_path: Path,
        output_dir: Path,
        mode: str = "single_view",
        initial_report: Optional[Dict[str, Any]] = None,
    ) -> ProjectRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self.SessionLocal() as session:
            rec = ProjectRecord(
                project_id=project_id,
                name=name,
                mode=mode,
                status="uploaded",
                progress_pct=5,
                stage_message="Image validated and preprocessed",
                source_image_path=str(source_image_path.resolve()),
                output_dir=str(output_dir.resolve()),
                parameters_json="{}",
                report_json=json.dumps(initial_report) if initial_report else None,
                error_message=None,
                created_at=now,
                updated_at=now,
            )
            merged = session.merge(rec)
            session.commit()
            session.refresh(merged)
            return merged

    def update_project_status(
        self,
        project_id: str,
        status: str,
        progress_pct: int,
        stage_message: str,
        parameters: Optional[Dict[str, Any]] = None,
        report: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Optional[ProjectRecord]:
        now = datetime.now(timezone.utc).isoformat()
        with self.SessionLocal() as session:
            rec = session.get(ProjectRecord, project_id)
            if rec is None:
                return None
            rec.status = status
            rec.progress_pct = int(progress_pct)
            rec.stage_message = stage_message
            if parameters is not None:
                rec.parameters_json = json.dumps(parameters)
            if report is not None:
                rec.report_json = json.dumps(report)
            rec.error_message = error_message
            rec.updated_at = now
            session.commit()
            session.refresh(rec)
            return rec

    def get_project(self, project_id: str) -> Optional[ProjectRecord]:
        with self.SessionLocal() as session:
            return session.get(ProjectRecord, project_id)

    def list_projects(self) -> List[ProjectRecord]:
        with self.SessionLocal() as session:
            return (
                session.query(ProjectRecord)
                .order_by(ProjectRecord.updated_at.desc())
                .all()
            )
