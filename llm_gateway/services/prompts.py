"""Prompt template service for managing and rendering prompt templates."""
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from llm_gateway.utils import get_logger

logger = get_logger(__name__)

# Singleton instance
_prompt_service = None


def get_prompt_service():
    """Get the global prompt service instance."""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptService()
    return _prompt_service


class PromptService:
    """Service for managing and rendering prompt templates."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize the prompt service.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates: Dict[str, str] = {}
        self.templates_dir = templates_dir or os.environ.get(
            "PROMPT_TEMPLATES_DIR", 
            str(Path.home() / ".llm_gateway" / "templates")
        )
        
        # Create templates directory if it doesn't exist
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Read templates from files
        self._read_templates()
        logger.info(f"Prompt service initialized with {len(self.templates)} templates")
    
    def _read_templates(self) -> None:
        """Read templates from files in the templates directory."""
        try:
            template_files = list(Path(self.templates_dir).glob("*.json"))
            logger.info(f"Found {len(template_files)} template files")
            
            for template_file in template_files:
                try:
                    with open(template_file, "r", encoding="utf-8") as f:
                        templates_data = json.load(f)
                    
                    # Add templates from file
                    for template_name, template_content in templates_data.items():
                        if isinstance(template_content, str):
                            self.templates[template_name] = template_content
                        elif isinstance(template_content, dict) and "text" in template_content:
                            self.templates[template_name] = template_content["text"]
                        else:
                            logger.warning(f"Invalid template format for {template_name}")
                            
                    logger.info(f"Loaded templates from {template_file.name}")
                except Exception as e:
                    logger.error(f"Error loading template file {template_file.name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading templates: {str(e)}")
    
    def _save_templates(self) -> None:
        """Save all templates to disk."""
        try:
            # Group templates by category
            categorized_templates: Dict[str, Dict[str, Any]] = {}
            
            for template_name, template_text in self.templates.items():
                # Extract category from template name (before first _)
                parts = template_name.split("_", 1)
                category = parts[0] if len(parts) > 1 else "general"
                
                if category not in categorized_templates:
                    categorized_templates[category] = {}
                
                categorized_templates[category][template_name] = template_text
            
            # Save each category to its own file
            for category, templates in categorized_templates.items():
                file_path = Path(self.templates_dir) / f"{category}_templates.json"
                
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(templates, f, indent=2)
                
                logger.info(f"Saved {len(templates)} templates to {file_path.name}")
                
        except Exception as e:
            logger.error(f"Error saving templates: {str(e)}")
    
    def get_template(self, template_name: str) -> Optional[str]:
        """Get a prompt template by name.
        
        Args:
            template_name: Template name
            
        Returns:
            Template text or None if not found
        """
        return self.templates.get(template_name)
    
    def get_all_templates(self) -> Dict[str, str]:
        """Get all templates.
        
        Returns:
            Dictionary of template name to template text
        """
        return self.templates.copy()
    
    def register_template(self, template_name: str, template_text: str) -> bool:
        """Register a new template or update an existing one.
        
        Args:
            template_name: Template name
            template_text: Template text
            
        Returns:
            True if successful
        """
        try:
            self.templates[template_name] = template_text
            
            # Schedule template save
            asyncio.create_task(self._async_save_templates())
            
            return True
        except Exception as e:
            logger.error(f"Error registering template {template_name}: {str(e)}")
            return False
    
    async def _async_save_templates(self) -> None:
        """Save templates asynchronously."""
        self._save_templates()
    
    def remove_template(self, template_name: str) -> bool:
        """Remove a template.
        
        Args:
            template_name: Template name
            
        Returns:
            True if removed, False if not found
        """
        if template_name in self.templates:
            del self.templates[template_name]
            
            # Schedule template save
            asyncio.create_task(self._async_save_templates())
            
            return True
        return False
    
    def render_template(
        self, 
        template_name: str, 
        variables: Dict[str, Any]
    ) -> Optional[str]:
        """Render a template with variables.
        
        Args:
            template_name: Template name
            variables: Variables to substitute
            
        Returns:
            Rendered template or None if error
        """
        template = self.get_template(template_name)
        if not template:
            logger.warning(f"Template {template_name} not found")
            return None
        
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in template {template_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {str(e)}")
            return None 