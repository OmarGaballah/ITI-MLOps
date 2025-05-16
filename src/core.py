from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dotenv import load_dotenv
from loguru import logger as loguru_logger

# ---------------------------
# Logger Class
# ---------------------------
class ExecutorLogger:
    def __init__(self, logs_path: Optional[str] = None, level: str = "INFO"):
        self.logger = loguru_logger
        self.logger.remove()  # Remove default
        self.logger.add(sys.stdout, level=level, format=self._get_console_format())

        if logs_path:
            log_file = Path("logs") / logs_path / f"logs_{datetime.now().strftime('%Y%m%d')}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.logger.add(log_file, level=level, format=self._get_file_format(), rotation="10 MB")

    @staticmethod
    def _get_console_format() -> str:
        return "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"

    @staticmethod
    def _get_file_format() -> str:
        return "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}"

    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def success(self, msg: str, *args, **kwargs):
        self.logger.success(msg, *args, **kwargs)

# ---------------------------
# Hydra Configuration Schema
# ---------------------------
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class PipelineConfig:
    drop: List[str] = field(default_factory=lambda: ["PassengerId", "Name", "Ticket", "Cabin"])
    imputation: Dict[str, List[str]] = field(default_factory=lambda: {"mean": ["Age"]})
    scaling: Dict[str, List[str]] = field(default_factory=lambda: {
        "standard": ["Age", "Fare"],
        "minmax": ["FamilySize", "TicketGroupSize"]
    })
    encoding: Dict[str, List[str]] = field(default_factory=lambda: {
        "onehot": ["Sex"],
        "ordinal": ["Embarked", "Deck", "Title"]
    })

@dataclass
class GlobalConfig:
    logs_path: Optional[str] = "globals"
    log_level: str = "INFO"
    pipeline_config: PipelineConfig = PipelineConfig()

# Register config with Hydra
cs = ConfigStore.instance()
cs.store(name="globals_config", node=GlobalConfig)

# ---------------------------
# Main (used in other scripts)
# ---------------------------
@hydra.main(config_path=None, config_name="globals_config", version_base="1.3")
def init_globals(cfg: GlobalConfig) -> None:
    load_dotenv()

    global logger, PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR
    global PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR
    global REPORTS_DIR, FIGURES_DIR, PIPELINE_CONFIG

    logger = ExecutorLogger(logs_path=cfg.logs_path, level=cfg.log_level)

    PROJ_ROOT = Path(hydra.utils.get_original_cwd())
    logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

    DATA_DIR = PROJ_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    INTERIM_DATA_DIR = DATA_DIR / "interim"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"

    MODELS_DIR = PROJ_ROOT / "models"

    REPORTS_DIR = PROJ_ROOT / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"

    PIPELINE_CONFIG = OmegaConf.to_container(cfg.pipeline_config, resolve=True)

    # Optional: tqdm integration
    try:
        from tqdm import tqdm
        logger.logger.remove(0)
        logger.logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    except ValueError:
        pass

if __name__ == "__main__":
    init_globals()
