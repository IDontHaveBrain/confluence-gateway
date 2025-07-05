"""Attachment generators for dummy data creation"""

import io
import os
import random
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .metadata import ATTACHMENT_METADATA

from .pdf_generator import PDFGenerator
from .office_generator import OfficeGenerator
from .image_generator import ImageGenerator

logger = logging.getLogger(__name__)

class AttachmentGenerator:
    """Main attachment generator that orchestrates different file types"""
    
    def __init__(self, client, config):
        self.client = client
        self.config = config

        self.generators = {
            "pdf": PDFGenerator(config),
            "docx": OfficeGenerator(config, file_type="docx"),
            "xlsx": OfficeGenerator(config, file_type="xlsx"),
            "png": ImageGenerator(config, file_type="png"),
            "jpg": ImageGenerator(config, file_type="jpg")
        }

        self.total_size_bytes = 0
        self.max_size_bytes = config.attachments.get("total_size_limit_mb", 100) * 1024 * 1024
    
    def create_attachment(self, page_id: str, category: str) -> bool:
        """Create and attach a file to a page"""
        if self.total_size_bytes >= self.max_size_bytes:
            logger.warning(f"Total attachment size limit reached ({self.config.attachments['total_size_limit_mb']}MB)")
            return False

        attachment_types = self.config.attachments.get("types", ["pdf"])
        logger.debug(f"Available attachment types: {attachment_types}")
        if not attachment_types:
            logger.error("No attachment types configured, using default 'pdf'")
            attachment_types = ["pdf"]
        file_type = random.choice(attachment_types)
        generator = self.generators.get(file_type)
        
        if not generator:
            logger.warning(f"No generator for file type: {file_type}")
            return False
        
        try:

            file_data = generator.generate(category)

            if file_data["size"] + self.total_size_bytes > self.max_size_bytes:
                logger.info("Skipping attachment - would exceed size limit")
                return False

            success = self._upload_attachment(
                page_id,
                file_data["filename"],
                file_data["content"],
                file_data["content_type"]
            )
            
            if success:
                self.total_size_bytes += file_data["size"]
                logger.info(f"Attached {file_data['filename']} ({file_data['size']/1024:.1f}KB) to page {page_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to create attachment: {e}")
            return False
    
    def _upload_attachment(
        self,
        page_id: str,
        filename: str,
        content: bytes,
        content_type: str
    ) -> bool:
        """Upload attachment to Confluence page"""
        try:

            result = self.client.atlassian_api.attach_content(
                content=content,
                name=filename,
                content_type=content_type,
                page_id=page_id,
                comment=f"Generated test attachment - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to upload attachment {filename}: {e}")
            return False

