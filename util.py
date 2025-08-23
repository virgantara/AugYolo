import torch
import pandas as pd
import os
from openpyxl import load_workbook

def top_k_accuracy(output, target, k=1):
    """Compute the top-k accuracy, but ensure k ≤ number of classes"""
    with torch.no_grad():
        max_k = min(k, output.size(1))  # Prevent k > number of classes
        _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct[:, :max_k].reshape(-1).float().sum(0).item()


def append_row_to_excel(path, row_dict, sheet_name="Sheet1"):
    """
    Appends a single row (dict) to an xlsx file.
    - Creates file with header if it doesn't exist.
    - If exists, appends below the last row.
    """
    df_row = pd.DataFrame([row_dict])

    if not os.path.exists(path):
        # New file: write with header
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_row.to_excel(writer, index=False, sheet_name=sheet_name)
        return

    # File exists: append without duplicating header
    book = load_workbook(path)
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        writer.book = book
        if sheet_name in writer.book.sheetnames:
            ws = writer.book[sheet_name]
            startrow = ws.max_row
        else:
            # create new sheet with header
            df_row.to_excel(writer, index=False, sheet_name=sheet_name)
            return
        df_row.to_excel(writer, index=False, header=False, sheet_name=sheet_name, startrow=startrow)
