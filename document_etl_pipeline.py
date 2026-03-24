"""
Project: Data RAG System
Module: Document ETL Pipeline
Description: 
    - 하이브리드 문서(PDF, DOCX, PPTX, XLSX)의 구조적 추출 및 마크다운 변환
    - 재귀적 디렉토리 스캔 및 계층 구조 보존 적재
    - 다국어(KO, EN) OCR 및 시각적 레이아웃 분석 지원
    - Memory & VRAM 최적화 적용 (Thread Pool & Worker Limit)
Author: H4RU
Date: 2026-03-17
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set, Optional, Iterator
# 메모리 누수 방지를 위해 multiprocessing Pool 사용
import multiprocessing
from tqdm import tqdm

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat

import traceback
import zipfile

# 전역 로거 세팅
logger = logging.getLogger("RAG_ETL")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', '%H:%M:%S')
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

class DocProcessingError(Exception):
    """문서 변환 실패 예외"""
    pass

@dataclass(frozen=True)
class ETLConfig:
    # 파이프라인 불변 설정
    input_dir: Path
    output_dir: Path
    # VRAM/RAM OOM 방지
    max_workers: int = 4
    target_exts: Set[str] = field(default_factory=lambda: {'.pdf', '.docx', '.pptx', '.xlsx'})
    skip_exts: Set[str] = field(default_factory=lambda: {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'})
    allowed_formats: List[InputFormat] = field(
        default_factory=lambda: [InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.XLSX]
    )

class ArchivePreprocessor:
    """ZIP 파일 탐색 및 자동 압축 해제"""
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir

    def run(self) -> None:
        zip_files = list(self.target_dir.rglob('*.zip'))
        if not zip_files:
            return
            
        logger.info(f"ZIP Archive 발견: {len(zip_files)}ea. Extract 시작")
        
        for zf in zip_files:
            if not zf.is_file():
                continue
                
            extract_dir = zf.with_suffix('') 
            
            if extract_dir.exists():
                continue
                
            try:
                with zipfile.ZipFile(zf, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                logger.debug(f"Unzip 성공: {zf.name}")
            except zipfile.BadZipFile:
                logger.error(f"ZIP file 손상: {zf.name}")
            except Exception as e:
                logger.error(f"Unzip 에러 [{zf.name}]: {e}")

class IFileScanner(ABC):
    """스캐너 인터페이스"""
    @abstractmethod
    def get_targets(self) -> Iterator[Path]:
        pass

class RecursiveScanner(IFileScanner):
    def __init__(self, config: ETLConfig):
        self.cfg = config

    def get_targets(self) -> Iterator[Path]:
        if not self.cfg.input_dir.exists():
            raise FileNotFoundError(f"경로 없음: {self.cfg.input_dir}")
            
        for f in self.cfg.input_dir.rglob('*'):
            if not f.is_file():
                continue
                
            # OS 임시 파일 및 숨김 파일 스킵
            if f.name.startswith('~$') or f.name.startswith('.'):
                continue

            ext = f.suffix.lower()
            if ext in self.cfg.target_exts:
                yield f
            elif ext in self.cfg.skip_exts:
                pass

# 워커 프로세스별 전역 캐시
_worker_converter = None

def _init_worker():
    """워커 프로세스 초기화 시 엔진 로드 방지 (지연 로딩 활용)"""
    global _worker_converter
    _worker_converter = None

def _process_route(args):
    """multiprocessing.Pool에서 호출하기 위한 최상위 경로 함수"""
    fpath, input_dir, output_dir, allowed_formats = args
    return DocumentETL._global_process_single(fpath, input_dir, output_dir, allowed_formats)

class DocumentETL:
    """문서 추출 및 변환 코어 엔진"""
    def __init__(self, config: ETLConfig, scanner: IFileScanner):
        self.cfg = config
        self.scanner = scanner
        # 메인 프로세스에서는 엔진을 로드하지 않음 (메모리 절약)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ETL 파이프라인 로드 완료 (Workers: {self.cfg.max_workers})")

    @staticmethod
    def _build_engine_static(allowed_formats) -> DocumentConverter:
        # 초기화
        ocr_opts = EasyOcrOptions(lang=["ko", "en"])
        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = True
        pdf_opts.ocr_options = ocr_opts
        pdf_opts.do_table_structure = True
        
        # 옵션 명시적 매핑 (Dictionary)
        return DocumentConverter(
            allowed_formats=allowed_formats,
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
            }
        )

    @staticmethod
    def _global_process_single(fpath: Path, input_dir: Path, output_dir: Path, allowed_formats) -> Optional[Path]:
        global _worker_converter
        start_t = time.time()
        size_mb = fpath.stat().st_size / (1024 * 1024)

        try:
            # [기능 추가] 이어하기 체크 로직
            rel_path = fpath.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix('.md')
            
            if out_path.exists():
                return out_path

            # 개별 워커 프로세스에서 엔진 최초 1회 로드
            if _worker_converter is None:
                _worker_converter = DocumentETL._build_engine_static(allowed_formats)

            logger.debug(f"처리 시작: {fpath.name}")
            res = _worker_converter.convert(fpath)
            md_text = res.document.export_to_markdown()
            
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md_text, encoding="utf-8")
            
            elapsed = time.time() - start_t
            speed = size_mb / elapsed if elapsed > 0 else 0
            
            logger.debug(f"파싱 성공: {fpath.name} ({elapsed:.2f}s, {speed:.2f} mb/s)")
            return out_path
            
        except Exception as e:
            # 에러 Traceback 전체 출력
            logger.error(f"Parse failed [{fpath.name}]: {e}\n{traceback.format_exc()}")
            return None

    def execute(self) -> None:
        targets = list(self.scanner.get_targets())
        if not targets:
            logger.warning("처리 대상 없음")
            return

        logger.info(f"배치 실행: 총 {len(targets)}건")
        start_t = time.time()
        success_cnt = 0

        # Memory Leak 방지를 위해 maxtasksperchild 설정 적용
        # 각 프로세스는 50개 파일 처리 후 자동 종료 및 재생성되어 메모리를 반환함
        task_args = [
            (f, self.cfg.input_dir, self.cfg.output_dir, self.cfg.allowed_formats) 
            for f in targets
        ]

        with multiprocessing.Pool(
            processes=self.cfg.max_workers,
            initializer=_init_worker,
            maxtasksperchild=50
        ) as pool:
            # imap_unordered를 사용하여 실시간 프로그레스 바 연동
            for result in tqdm(pool.imap_unordered(_process_route, task_args), total=len(targets), desc="ETL Progress"):
                if result:
                    success_cnt += 1

        total_t = time.time() - start_t
        logger.info(f"배치 완료: {success_cnt}/{len(targets)} 성공 (총 {total_t:.1f}s)")

if __name__ == "__main__":
    # 설정 및 의존성 주입 (Dependency Injection)
    cfg = ETLConfig(
        input_dir=Path("./자료"),
        output_dir=Path("./processed_md")
    )

    preprocessor = ArchivePreprocessor(target_dir=cfg.input_dir)
    preprocessor.run()

    scanner = RecursiveScanner(config=cfg)
    etl = DocumentETL(config=cfg, scanner=scanner)
    
    etl.execute()