import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def extractRestaurantDetailsFromReview(sample, search_words=None, verbose=False):
    # Takes a review text and applies regex to extract specific details 
    # (like service, price range, food score) based on the provided search patterns.
    
    clean_text = re.sub(r'\\ue[0-9a-f]{3}', '', sample)
    clean_text = re.sub(r'\n+', '\n', clean_text)
    clean_text = clean_text.strip()

    # Store extracted values
    extracted_values = []

    # Loop through search words to extract values dynamically
    for key, regex in search_words.items():
        match = re.search(regex, clean_text)
        value = match.group(1) if match else ''
        extracted_values.append(value)

    return extracted_values

def applyExtractDetails(df, search_words=None):
    # Applies the extraction function to the entire DataFrame, creating new columns 
    # for the extracted details based on the regex patterns provided.

    column_names = list(search_words.keys())
    df[column_names] = df['text_backup'].apply(lambda x: pd.Series(extractRestaurantDetailsFromReview(x, search_words=search_words)))
    return df

def extractReviewCount(text):
    # Extracts the number of reviews from a string (Spanish "reseñas" or English "reviews").

    if isinstance(text, str):
        match = re.search(r'(\d+)\s+(?:reseñas|reviews)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def extractStarRating(text):
    # Extracts the star rating (out of 5) from a review string (Spanish "estrellas" or English "stars").

    if isinstance(text, str):
        match = re.search(r'(\d+)\s+(?:estrellas|stars)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def extractRecommendations(recommendations):
    # Splits a list of recommended dishes; handles " and " (English) and " y " (Spanish).

    recommendations_list = recommendations.split(', ')
    if recommendations_list:
        last = recommendations_list[-1]
        if ' and ' in last:
            parts = last.rsplit(' and ', 1)
            recommendations_list = recommendations_list[:-1] + [p.strip() for p in parts]
        elif ' y ' in last:
            parts = last.rsplit(' y ', 1)
            recommendations_list = recommendations_list[:-1] + [p.strip() for p in parts]
    return recommendations_list

def convertToDate(date_text):
    # Converts relative date (e.g. "2 weeks ago", "3 months ago") to an exact date.
    # Supports English (week/month/year) and Spanish (semana/mes/año).

    if not isinstance(date_text, str):
        return None
    date_text_lower = date_text.lower()
    today = datetime.today()

    # Weeks: "semana" (Spanish) or "week" (English)
    if 'semana' in date_text_lower or 'week' in date_text_lower:
        weeks = pd.Series(date_text).str.extract(r'(\d+)')[0]
        weeks = int(weeks.iloc[0]) if pd.notna(weeks.iloc[0]) else 1
        monday_of_current_week = today - timedelta(days=today.weekday())
        return monday_of_current_week.date() - timedelta(weeks=weeks)

    # Months: "mes" (Spanish) or "month" (English)
    if 'mes' in date_text_lower or 'month' in date_text_lower:
        months = pd.Series(date_text).str.extract(r'(\d+)')[0]
        months = int(months.iloc[0]) if pd.notna(months.iloc[0]) else 1
        target_date = today - relativedelta(months=months)
        return target_date.replace(day=1).date()

    # Years: "año" (Spanish) or "year" (English)
    if 'año' in date_text_lower or 'year' in date_text_lower:
        years = pd.Series(date_text).str.extract(r'(\d+)')[0]
        years = int(years.iloc[0]) if pd.notna(years.iloc[0]) else 1
        target_date = today - relativedelta(years=years)
        return target_date.replace(month=1, day=1).date()

    return None
