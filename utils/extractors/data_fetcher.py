# utils/data_fetcher.py
import json
import asyncio

import requests

import streamlit as st

from playwright.async_api import async_playwright
# Async Playwright JSON fetcher
async def fetch_json_data(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        pre_tag = await page.query_selector("pre")
        json_text = await pre_tag.text_content() if pre_tag else "{}"
        await browser.close()
        return json.loads(json_text)

# Sync wrapper
def fetch_json(url):
    return asyncio.run(fetch_json_data(url))

# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def fetch_json(url):
#     try:
#         logger.info(f"Fetching URL: {url}")
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         logger.error(f"Error fetching {url}: {e}")
#         return {}

def fetch_seasons_json(tournament):
    seasons_url = f"https://api.sofascore.com/api/v1/unique-tournament/{tournament['unique_tournament']}/seasons"
    # https://api.sofascore.com/api/v1/unique-tournament/17/seasons
    return fetch_json(seasons_url)

def fetch_standing_json(tournament, season):
    standings_url = (
            f"https://www.sofascore.com/api/v1/unique-tournament/"
            f"{tournament['unique_tournament']}/season/{season['id']}/standings/total"
        )
    # https://www.sofascore.com/api/v1/unique-tournament/17/season/61627/standings/total
    return fetch_json(standings_url)

def fetch_rounds_json(tournament, season):
    rounds_url = (
        f"https://www.sofascore.com/api/v1/unique-tournament/"
        f"{tournament['unique_tournament']}/season/{season['id']}/rounds"
    )
    # https://www.sofascore.com/api/v1/unique-tournament/17/season/61627/rounds"
    return fetch_json(rounds_url)

def fetch_round_events(tournament, season, round_number):
    round_events_url = (
        f"https://www.sofascore.com/api/v1/unique-tournament/"
        f"{tournament['unique_tournament']}/season/{season['id']}/events/round/{round_number}"
    )
    # https://www.sofascore.com/api/v1/unique-tournament/17/season/61627/events/round/1
    round_response = fetch_json(round_events_url)
    round_events = round_response.get("events", [])
    return round_events

def fetch_lineups(fixture_id):
    round_events_url = (
        f"https://www.sofascore.com/api/v1/event/"
        f"{fixture_id}/lineups"
    )
    # https://www.sofascore.com/api/v1/event/11352546/lineups
    round_response = fetch_json(round_events_url)
    round_events = round_response.get("confirmed", [])
    return round_events

def fetch_managers(fixture_id):
    managers_url = f"https://www.sofascore.com/api/v1/event/{fixture_id}/managers"
    # https://www.sofascore.com/api/v1/event/12436870/managers
    return fetch_json(managers_url)

def fetch_incidents(fixture_id):
    incidents_url = f"https://www.sofascore.com/api/v1/event/{fixture_id}/incidents"
    # https://www.sofascore.com/api/v1/event/12436870/incidents
    return fetch_json(incidents_url)

def fetch_shotmap(fixture_id):
    shotmap_url = f"https://www.sofascore.com/api/v1/event/{fixture_id}/shotmap"
    # https://www.sofascore.com/api/v1/event/12436870/shotmap
    return fetch_json(shotmap_url)