import re
from typing import List, Dict
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ReportParser:
    def __init__(self, model_path: str = "emilyalsentzer/Bio_ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=len(cfg.target_columns)
        )
        self.model.eval()
        self.device = cfg.device
        self.model.to(self.device)

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s\.,]", "", text)
        return text.strip()

    def extract_labels(self, reports: List[str]) -> torch.Tensor:
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
        df_out = df.copy()
        raw_reports = df_out[text_col].tolist()
        extracted_probs = self.extract_labels(raw_reports).numpy()
        
        for i, col in enumerate(cfg.target_columns):
            df_out[f"pseudo_{col}"] = extracted_probs[:, i]
            
        return df_out