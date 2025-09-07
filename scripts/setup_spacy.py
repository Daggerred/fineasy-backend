#!/usr/bin/env python3
"""
Script to download and setup spaCy models for NLP invoice generation
"""
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_spacy_model():
    """Download spaCy English model"""
    try:
        logger.info("Downloading spaCy English model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        logger.info("spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download spaCy model: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading spaCy model: {e}")
        return False


def test_spacy_model():
    """Test if spaCy model is working"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        # Test with sample text
        doc = nlp("Generate an invoice for John Doe, 10 units of Widget A at ₹100 each")
        
        logger.info("spaCy model test successful")
        logger.info(f"Entities found: {[(ent.text, ent.label_) for ent in doc.ents]}")
        return True
    except Exception as e:
        logger.error(f"spaCy model test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("Setting up spaCy for NLP invoice generation...")
    
    if download_spacy_model():
        if test_spacy_model():
            logger.info("spaCy setup completed successfully!")
        else:
            logger.error("spaCy setup failed - model test failed")
            sys.exit(1)
    else:
        logger.error("spaCy setup failed - model download failed")
        sys.exit(1)