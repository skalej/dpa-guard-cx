import importlib.util
import sys
import types
from pathlib import Path


def _ensure_module(name: str, module: types.ModuleType):
    if name not in sys.modules:
        sys.modules[name] = module


def _stub_dependencies():
    _ensure_module("boto3", types.ModuleType("boto3"))

    docx_mod = types.ModuleType("docx")
    docx_mod.Document = lambda *args, **kwargs: None
    _ensure_module("docx", docx_mod)

    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = object
    openai_mod.APIError = type("APIError", (Exception,), {})
    openai_mod.APIConnectionError = type("APIConnectionError", (Exception,), {})
    openai_mod.RateLimitError = type("RateLimitError", (Exception,), {})
    openai_mod.APITimeoutError = type("APITimeoutError", (Exception,), {})
    _ensure_module("openai", openai_mod)

    pgvector_mod = types.ModuleType("pgvector")
    pgvector_sqlalchemy = types.ModuleType("pgvector.sqlalchemy")
    pgvector_sqlalchemy.Vector = lambda *args, **kwargs: None
    _ensure_module("pgvector", pgvector_mod)
    _ensure_module("pgvector.sqlalchemy", pgvector_sqlalchemy)

    pypdf_mod = types.ModuleType("pypdf")
    pypdf_mod.PdfReader = lambda *args, **kwargs: None
    _ensure_module("pypdf", pypdf_mod)

    yaml_mod = types.ModuleType("yaml")
    yaml_mod.safe_load = lambda *args, **kwargs: None
    _ensure_module("yaml", yaml_mod)

    negotiation_mod = types.ModuleType("negotiation_templates")
    negotiation_mod.get_template = lambda *args, **kwargs: {}
    _ensure_module("negotiation_templates", negotiation_mod)

    celery_mod = types.ModuleType("celery")

    class _CeleryStub:
        def __init__(self, *args, **kwargs):
            pass

        def task(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def send_task(self, *args, **kwargs):
            return None

    celery_mod.Celery = _CeleryStub
    _ensure_module("celery", celery_mod)

    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    sqlalchemy_mod.Column = lambda *args, **kwargs: None
    sqlalchemy_mod.DateTime = lambda *args, **kwargs: None
    sqlalchemy_mod.Integer = lambda *args, **kwargs: None
    sqlalchemy_mod.String = lambda *args, **kwargs: None
    sqlalchemy_mod.Text = lambda *args, **kwargs: None
    sqlalchemy_mod.create_engine = lambda *args, **kwargs: None
    sqlalchemy_mod.delete = lambda *args, **kwargs: None
    sqlalchemy_mod.func = types.SimpleNamespace(now=lambda: None)
    sqlalchemy_mod.select = lambda *args, **kwargs: None
    _ensure_module("sqlalchemy", sqlalchemy_mod)

    sqlalchemy_dialects = types.ModuleType("sqlalchemy.dialects")
    sqlalchemy_pg = types.ModuleType("sqlalchemy.dialects.postgresql")
    sqlalchemy_pg.JSONB = lambda *args, **kwargs: None
    sqlalchemy_pg.UUID = lambda *args, **kwargs: None
    _ensure_module("sqlalchemy.dialects", sqlalchemy_dialects)
    _ensure_module("sqlalchemy.dialects.postgresql", sqlalchemy_pg)

    sqlalchemy_exc = types.ModuleType("sqlalchemy.exc")
    sqlalchemy_exc.IntegrityError = type("IntegrityError", (Exception,), {})
    _ensure_module("sqlalchemy.exc", sqlalchemy_exc)

    sqlalchemy_orm = types.ModuleType("sqlalchemy.orm")
    sqlalchemy_orm.Session = object
    sqlalchemy_orm.declarative_base = lambda *args, **kwargs: type("Base", (), {})
    sqlalchemy_orm.sessionmaker = lambda *args, **kwargs: lambda *a, **k: None
    _ensure_module("sqlalchemy.orm", sqlalchemy_orm)


def load_worker_app():
    _stub_dependencies()
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("worker_app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module
