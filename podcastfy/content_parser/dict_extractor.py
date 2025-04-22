"""
Dictornary Extractor Module

Designed as a general llm output flattener and parse 
Furthher development will see content generation specific connectors - slides and general structured output
"""
import os
import logging
import unicodedata

from typing import Iterable, List

logger = logging.getLogger(__name__)

class DictExtractor:

	def all_vals(self, obj):
		if isinstance(obj, dict):
			for v in obj.values():
				yield from self.all_vals(v)
		else:
			yield obj
		
	def extract_content(self, d: dict) -> List[str]:
		"""
		Extracts text content from a dictionary.

		Args:
			d (dict): Dictionary containing the content to extract.

		Returns:
			List[str]: List of extracted text content.
		"""
	
		try:
			content = list(self.all_vals(d))

			return content

		except Exception as e:
			logger.error(f"Error extracting PDF content: {str(e)}")
			raise
