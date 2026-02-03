"""Validation utilities for document uploads."""

import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.config import settings

logger = logging.getLogger(__name__)

# Maximum file size for document uploads (in bytes)
# Default: 50 MB, configurable via settings
MAX_DOCUMENT_SIZE = settings.MAX_DOCUMENT_UPLOAD_SIZE_MB * 1024 * 1024


async def validate_document_file(file: UploadFile) -> int:
    """
    Validate document file size.

    Args:
        file: The uploaded file

    Returns:
        File size in bytes if valid

    Raises:
        HTTPException: If validation fails
    """
    if not file.filename:
        raise HTTPException(
            status_code=422,
            detail="File must have a filename",
        )

    # Get file size by seeking to end
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_DOCUMENT_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_DOCUMENT_SIZE / (1024 * 1024):.0f}MB",
        )

    if file_size == 0:
        raise HTTPException(
            status_code=422,
            detail="Empty file uploaded"
        )

    logger.info(f"Document file validated: {file.filename} ({file_size} bytes)")
    return file_size
