import re
from typing import List
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import cfg

class ReportParser:
    """
    NLP Extractor using ClinicalBERT for processing radiology text reports
    and generating soft target label probabilities.
    """
    def __init__(self, model_path: str = "emilyalsentzer/Bio_ClinicalBERT"):
        self.device = cfg.device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                num_labels=len(cfg.target_columns)
            ).to(self.device)
            self.model.eval()
            self.is_ready = True
        except Exception as e:
            print(f"NLP Parser initialization warning: {e}. Report feature extraction bypassed.")
            self.is_ready = False

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s\.,]", "", text)
        return text.strip()

    def extract_labels(self, reports: List[str]) -> torch.Tensor:
        if not self.is_ready:
            return torch.zeros((len(reports), cfg.num_classes))
            
        cleaned_reports = [self.clean_text(r) for r in reports]
        inputs = self.tokenizer(
            cleaned_reports,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            
        return probs.cpu()

    def process_dataframe(self, df: pd.DataFrame, text_col: str = "report") -> pd.DataFrame:
        if text_col not in df.columns:
            return df
        df_out = df.copy()
        raw_reports = df_out[text_col].tolist()
        extracted_probs = self.extract_labels(raw_reports).numpy()
        
        for i, col in enumerate(cfg.target_columns):
            df_out[f"pseudo_{col}"] = extracted_probs[:, i]
            
        return df_out