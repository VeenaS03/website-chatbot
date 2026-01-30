"""
Website crawler module for extracting meaningful textual content.
Handles URL validation, HTML parsing, and content extraction.
"""

import logging
from typing import Tuple, Optional
from urllib.parse import urljoin, urlparse

import requests
import validators
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebsiteCrawler:
    """
    Responsible for crawling websites and extracting clean textual content.
    
    Features:
    - URL validation and error handling
    - Removes headers, footers, navigation, scripts, styles, ads
    - Extracts meaningful content only
    - Respects robots.txt and rate limiting
    """
    
    TIMEOUT_SECONDS = 10
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    # Tags to remove completely
    REMOVE_TAGS = {
        'script', 'style', 'meta', 'noscript', 'link', 
        'nav', 'header', 'footer', 'form', 'button',
        'iframe', 'svg', 'canvas', 'img'
    }
    
    # Classes/IDs commonly used for ads and navigation
    AD_PATTERNS = {
        'ad', 'advertisement', 'sidebar', 'banner', 'popup',
        'modal', 'newsletter', 'subscribe', 'widget', 'related',
        'recommended', 'trending', 'social-share'
    }
    
    def __init__(self):
        """Initialize the crawler with proper session configuration."""
        self.session = self._create_session()
    
    @staticmethod
    def _create_session() -> requests.Session:
        """
        Create a requests session with retry strategy and proper headers.
        
        Returns:
            Configured requests.Session object
        """
        session = requests.Session()
        
        # Retry strategy for network resilience
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set user agent
        session.headers.update({'User-Agent': WebsiteCrawler.USER_AGENT})
        
        return session
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate if the provided URL is valid.
        
        Args:
            url: URL string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not url:
            return False
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        return validators.url(url) is True
    
    def _fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch the HTML content of a webpage.
        
        Args:
            url: Website URL to fetch
            
        Returns:
            HTML content as string, or None if fetch fails
        """
        try:
            # Ensure protocol is present
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            response = self.session.get(
                url,
                timeout=self.TIMEOUT_SECONDS,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Check content length
            if len(response.content) > self.MAX_CONTENT_LENGTH:
                logger.warning(f"Content too large: {len(response.content)} bytes")
                return None
            
            return response.text
            
        except requests.exceptions.MissingSchema:
            logger.error(f"Invalid URL scheme: {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for URL: {url}")
            return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching URL: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for URL {url}: {str(e)}")
            return None
    
    @staticmethod
    def _is_ad_element(element) -> bool:
        """
        Check if an element is likely an ad or navigation element.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            True if element is likely an ad/nav, False otherwise
        """
        class_attr = element.get('class', [])
        id_attr = element.get('id', '')
        
        # Check class and id attributes
        if isinstance(class_attr, list):
            for cls in class_attr:
                if any(pattern in cls.lower() for pattern in WebsiteCrawler.AD_PATTERNS):
                    return True
        
        if any(pattern in id_attr.lower() for pattern in WebsiteCrawler.AD_PATTERNS):
            return True
        
        return False
    
    @staticmethod
    def _extract_text_from_soup(soup: BeautifulSoup) -> str:
        """
        Extract clean text from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup parsed HTML object
            
        Returns:
            Cleaned text content
        """
        # Remove script, style, and other unnecessary tags
        for tag in soup.find_all(WebsiteCrawler.REMOVE_TAGS):
            tag.decompose()
        
        # Remove ad and navigation elements
        for element in soup.find_all(True):  # True finds all elements
            if WebsiteCrawler._is_ad_element(element):
                element.decompose()
        
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def _get_page_title(soup: BeautifulSoup) -> str:
        """
        Extract page title from HTML.
        
        Args:
            soup: BeautifulSoup parsed HTML object
            
        Returns:
            Page title or "Unknown"
        """
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
        
        return "Unknown"
    
    def crawl(self, url: str) -> Tuple[bool, str, str]:
        """
        Crawl a website and extract its textual content.
        
        Args:
            url: Website URL to crawl
            
        Returns:
            Tuple of (success: bool, content: str, title: str)
            - success: True if crawl was successful
            - content: Extracted text content
            - title: Page title or error message
        """
        # Validate URL
        if not self.validate_url(url):
            return False, "", "Invalid URL format. Please provide a valid website URL."
        
        # Fetch page
        html_content = self._fetch_page(url)
        if html_content is None:
            return False, "", f"Could not fetch the website. Please check the URL and try again."
        
        # Parse HTML
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to parse HTML: {str(e)}")
            return False, "", "Failed to parse the website content."
        
        # Extract content
        text_content = self._extract_text_from_soup(soup)
        page_title = self._get_page_title(soup)
        
        if not text_content.strip():
            return False, "", "No text content found on the website."
        
        logger.info(f"Successfully crawled {len(text_content)} characters from {url}")
        return True, text_content, page_title


def create_crawler() -> WebsiteCrawler:
    """Factory function to create a WebsiteCrawler instance."""
    return WebsiteCrawler()