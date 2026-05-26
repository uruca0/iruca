import asyncio
import io
import json
import os
import re
import tempfile
import wave
from pathlib import Path
from typing import Optional

import discord
import enka
import numpy as np
import pyopenjtalk
from discord import app_commands
from discord.ext import commands
from PIL import Image, ImageDraw, ImageFont
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import aiohttp

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "YOUR_DISCORD_TOKEN_HERE")
DICT_FILE = Path("openjtalk_dict.json")
IGNORE_PREFIXES = ("!", "/", ".", "?")
SAMPLE_RATE = 48000

active_channels: dict = {}
read_queues: dict = {}
user_dict: dict = {}

STAT_NAMES = {
    "FIGHT_PROP_HP": "HP",
    "FIGHT_PROP_HP_PERCENT": "HP%",
    "FIGHT_PROP_ATTACK": "攻撃力",
    "FIGHT_PROP_ATTACK_PERCENT": "攻撃力%",
    "FIGHT_PROP_DEFENSE": "防御力",
    "FIGHT_PROP_DEFENSE_PERCENT": "防御力%",
    "FIGHT_PROP_ELEMENT_MASTERY": "元素熟知",
    "FIGHT_PROP_CRITICAL": "会心率",
    "FIGHT_PROP_CRITICAL_HURT": "会心ダメージ",
    "FIGHT_PROP_CHARGE_EFFICIENCY": "元素チャージ効率",
    "FIGHT_PROP_HEAL_ADD": "与える治癒効果",
    "FIGHT_PROP_FIRE_ADD_HURT": "炎元素ダメージ",
    "FIGHT_PROP_WATER_ADD_HURT": "水元素ダメージ",
    "FIGHT_PROP_WIND_ADD_HURT": "風元素ダメージ",
    "FIGHT_PROP_ELEC_ADD_HURT": "雷元素ダメージ",
    "FIGHT_PROP_ICE_ADD_HURT": "氷元素ダメージ",
    "FIGHT_PROP_ROCK_ADD_HURT": "岩元素ダメージ",
    "FIGHT_PROP_GRASS_ADD_HURT": "草元素ダメージ",
    "FIGHT_PROP_PHYSICAL_ADD_HURT": "物理ダメージ",
}

PERCENT_STATS = {
    "FIGHT_PROP_HP_PERCENT", "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_DEFENSE_PERCENT",
    "FIGHT_PROP_CRITICAL", "FIGHT_PROP_CRITICAL_HURT", "FIGHT_PROP_CHARGE_EFFICIENCY",
    "FIGHT_PROP_HEAL_ADD", "FIGHT_PROP_FIRE_ADD_HURT", "FIGHT_PROP_WATER_ADD_HURT",
    "FIGHT_PROP_WIND_ADD_HURT", "FIGHT_PROP_ELEC_ADD_HURT", "FIGHT_PROP_ICE_ADD_HURT",
    "FIGHT_PROP_ROCK_ADD_HURT", "FIGHT_PROP_GRASS_ADD_HURT", "FIGHT_PROP_PHYSICAL_ADD_HURT",
}

ARTIFACT_SLOT_NAMES = {
    "EQUIP_BRACER": "花",
    "EQUIP_NECKLACE": "羽",
    "EQUIP_SHOES": "砂",
    "EQUIP_RING": "杯",
    "EQUIP_DRESS": "冠",
}


def load_dict():
    global user_dict
    if DICT_FILE.exists():
        with open(DICT_FILE, encoding="utf-8") as f:
            user_dict = json.load(f)


def save_dict():
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(user_dict, f, ensure_ascii=False, indent=2)


def apply_dict(text: str) -> str:
    for word, reading in user_dict.items():
        text = text.replace(word, reading)
    return text


def soft_limit(x: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    signs = np.sign(x)
    abs_x = np.abs(x)
    over = abs_x > threshold
    abs_x[over] = threshold + (1.0 - threshold) * np.tanh(
        (abs_x[over] - threshold) / (1.0 - threshold)
    )
    return signs * abs_x


def synthesize_sync(text: str) -> Optional[bytes]:
    try:
        wave_data, sr = pyopenjtalk.tts(text)
        peak = np.max(np.abs(wave_data))
        if peak > 0:
            wave_data = wave_data / peak
        wave_data = soft_limit(wave_data, threshold=0.8)
        wave_data = wave_data * 0.85
        wave_int16 = (wave_data * 32767).clip(-32768, 32767).astype(np.int16)
        if sr != SAMPLE_RATE:
            new_length = int(len(wave_int16) * SAMPLE_RATE / sr)
            indices = np.linspace(0, len(wave_int16) - 1, new_length)
            wave_int16 = np.interp(indices, np.arange(len(wave_int16)), wave_int16).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(wave_int16.tobytes())
        return buf.getvalue()
    except Exception as e:
        print(f"[OpenJTalk] 合成エラー: {e}")
        return None


async def synthesize(text: str) -> Optional[bytes]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, synthesize_sync, text)


def format_stat_value(prop_id: str, value: float) -> str:
    if prop_id in PERCENT_STATS:
        return f"{value * 100:.1f}%"
    return f"{int(value)}"


async def fetch_image(session: aiohttp.ClientSession, url: str) -> Optional[Image.Image]:
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.read()
                return Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:
        pass
    return None


def build_card_image(character, player, weights: dict = None, score_type: str = "攻撃換算") -> io.BytesIO:
    if weights is None:
        weights = SCORE_WEIGHTS["攻撃換算"]
    W, H = 900, 540
    card = Image.new("RGBA", (W, H), (20, 20, 30, 255))
    draw = ImageDraw.Draw(card)

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_mid   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font_large = ImageFont.load_default()
        font_mid   = font_large
        font_small = font_large

    # ヘッダー背景
    draw.rectangle([(0, 0), (W, 70)], fill=(40, 40, 60, 255))

    # キャラクター名・レベル
    char_name = getattr(character, "name", "Unknown")
    char_level = getattr(character, "level", "?")
    constellation = getattr(character, "constellations_unlocked", 0)
    draw.text((20, 10), str(char_name), font=font_large, fill=(255, 220, 100))
    draw.text((20, 44), f"Lv.{char_level}  C{constellation}", font=font_mid, fill=(200, 200, 200))

    # プレイヤー名
    player_name = getattr(player, "nickname", "")
    draw.text((W - 20, 10), str(player_name), font=font_small, fill=(180, 180, 180), anchor="ra")

    # 武器情報
    weapon = getattr(character, "weapon", None)
    y = 90
    draw.text((20, y), "── 武器 ──", font=font_mid, fill=(150, 200, 255))
    y += 28
    if weapon:
        w_name  = getattr(weapon, "name", "不明")
        w_level = getattr(weapon, "level", "?")
        w_ref   = getattr(weapon, "refinement", 1)
        draw.text((20, y), f"{w_name}  Lv.{w_level}  精錬{w_ref}", font=font_small, fill=(230, 230, 230))
    y += 26

    # ステータス
    draw.text((20, y), "── ステータス ──", font=font_mid, fill=(150, 200, 255))
    y += 28
    stats = getattr(character, "stats", None)
    priority_props = [
        "FIGHT_PROP_HP", "FIGHT_PROP_ATTACK", "FIGHT_PROP_DEFENSE",
        "FIGHT_PROP_ELEMENT_MASTERY", "FIGHT_PROP_CRITICAL", "FIGHT_PROP_CRITICAL_HURT",
        "FIGHT_PROP_CHARGE_EFFICIENCY",
    ]
    col = 0
    for prop_id in priority_props:
        if stats is None:
            break
        try:
            val = getattr(stats, prop_id.lower().replace("fight_prop_", ""), None)
            if val is None:
                continue
            label = STAT_NAMES.get(prop_id, prop_id)
            text_val = format_stat_value(prop_id, float(val))
            x = 20 + (col % 2) * 420
            draw.text((x, y), f"{label}: {text_val}", font=font_small, fill=(220, 220, 220))
            if col % 2 == 1:
                y += 20
            col += 1
        except Exception:
            continue
    if col % 2 == 1:
        y += 20
    y += 10

    # 聖遺物
    artifacts = getattr(character, "artifacts", [])
    if artifacts:
        draw.text((20, y), "── 聖遺物 ──", font=font_mid, fill=(150, 200, 255))
        y += 28
        total_score = 0.0
        for art in artifacts:
            slot_raw  = getattr(art, "equip_type", "")
            slot_name = ARTIFACT_SLOT_NAMES.get(str(slot_raw), str(slot_raw))
            art_name  = getattr(art, "name", "不明")
            art_level = getattr(art, "level", "?")
            main_stat = getattr(art, "main_stat", None)
            main_label = ""
            if main_stat:
                ms_id  = str(getattr(main_stat, "prop_id", ""))
                ms_val = getattr(main_stat, "value", 0)
                main_label = f"[{STAT_NAMES.get(ms_id, ms_id)} {format_stat_value(ms_id, float(ms_val))}]"
            art_score = calc_artifact_score(art, weights)
            total_score += art_score
            score_str = f"  スコア:{art_score:.1f}"
            draw.text((20, y), f"{slot_name} {art_name} +{art_level} {main_label}{score_str}", font=font_small, fill=(210, 210, 210))
            y += 18
            if y > H - 30:
                break
        draw.text((20, y), f"合計スコア: {total_score:.1f}（{score_type}）", font=font_mid, fill=(255, 220, 100))
        y += 24

    # 枠線
    draw.rectangle([(0, 0), (W - 1, H - 1)], outline=(80, 80, 120), width=2)

    buf = io.BytesIO()
    card.save(buf, format="PNG")
    buf.seek(0)
    return buf


intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree


async def queue_worker(guild: discord.Guild):
    queue = read_queues[guild.id]
    while True:
        text = await queue.get()
        vc = guild.voice_client
        if vc is None or not vc.is_connected():
            queue.task_done()
            continue
        try:
            wav = await synthesize(text)
            if wav is None:
                queue.task_done()
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav)
                tmp_path = tmp.name
            while vc.is_playing():
                await asyncio.sleep(0.1)
            vc.play(
                discord.FFmpegPCMAudio(tmp_path),
                after=lambda e: os.unlink(tmp_path) if e is None else None,
            )
            while vc.is_playing():
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] 読み上げ失敗: {e}")
        finally:
            queue.task_done()


@bot.event
async def on_ready():
    load_dict()
    await tree.sync()
    print(f"Logged in as {bot.user}")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    guild = message.guild
    if guild is None:
        return
    if active_channels.get(guild.id) != message.channel.id:
        return
    if message.content.startswith(IGNORE_PREFIXES):
        return
    text = message.content.strip()
    if not text:
        return
    text = re.sub(r"https?://\S+", "URL省略", text)
    text = re.sub(r"<@!?\d+>", "", text)
    text = re.sub(r"<a?:\w+:\d+>", "", text)
    text = text.strip()
    if not text:
        return
    if len(text) > 100:
        text = text[:100] + "、以下省略"
    text = apply_dict(text)
    if guild.id in read_queues:
        await read_queues[guild.id].put(text)
    await bot.process_commands(message)


@tree.command(name="読み上げ開始", description="このチャンネルの読み上げを開始します")
async def start_reading(interaction: discord.Interaction):
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message("サーバー内でのみ使用できます。", ephemeral=True)
        return
    member = interaction.user
    if not isinstance(member, discord.Member) or member.voice is None:
        await interaction.response.send_message("先にボイスチャンネルに参加してください。", ephemeral=True)
        return
    vc_channel = member.voice.channel
    if guild.voice_client is not None:
        await guild.voice_client.disconnect()
    try:
        await vc_channel.connect()
    except Exception as e:
        await interaction.response.send_message(f"VCへの接続に失敗しました: {e}", ephemeral=True)
        return
    active_channels[guild.id] = interaction.channel_id
    if guild.id not in read_queues:
        read_queues[guild.id] = asyncio.Queue()
        bot.loop.create_task(queue_worker(guild))
    await interaction.response.send_message(
        f"🔊 **読み上げ開始！**\nテキストチャンネル: {interaction.channel.mention}\nボイスチャンネル: **{vc_channel.name}**"
    )
    await read_queues[guild.id].put("読み上げを開始します")


@tree.command(name="読み上げ終了", description="読み上げを停止してVCから退出します")
async def stop_reading(interaction: discord.Interaction):
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message("サーバー内でのみ使用できます。", ephemeral=True)
        return
    if guild.id not in active_channels:
        await interaction.response.send_message("現在、読み上げは有効になっていません。", ephemeral=True)
        return
    active_channels.pop(guild.id, None)
    if guild.id in read_queues:
        while not read_queues[guild.id].empty():
            try:
                read_queues[guild.id].get_nowait()
                read_queues[guild.id].task_done()
            except asyncio.QueueEmpty:
                break
    if guild.voice_client is not None:
        await guild.voice_client.disconnect()
    await interaction.response.send_message("🔇 **読み上げを終了しました。**")


@tree.command(name="辞書登録", description="読み替え辞書に単語を登録します")
@app_commands.describe(word="登録する単語", reading="読み替え後のテキスト")
async def register_dict(interaction: discord.Interaction, word: str, reading: str):
    if not word or not reading:
        await interaction.response.send_message("単語と読み方を入力してください。", ephemeral=True)
        return
    user_dict[word] = reading
    save_dict()
    await interaction.response.send_message(f"📖 辞書に登録しました\n`{word}` → `{reading}`")


@tree.command(name="辞書一覧", description="登録されている辞書の一覧を表示します")
async def show_dict(interaction: discord.Interaction):
    if not user_dict:
        await interaction.response.send_message("辞書に登録された単語はありません。", ephemeral=True)
        return
    lines = [f"`{word}` → `{reading}`" for word, reading in user_dict.items()]
    chunk = ""
    chunks = []
    for line in lines:
        if len(chunk) + len(line) + 1 > 1900:
            chunks.append(chunk)
            chunk = line
        else:
            chunk = (chunk + "\n" + line).strip()
    if chunk:
        chunks.append(chunk)
    await interaction.response.send_message(f"📖 **辞書一覧**\n{chunks[0]}", ephemeral=True)
    for extra in chunks[1:]:
        await interaction.followup.send(extra, ephemeral=True)


@tree.command(name="辞書削除", description="辞書から単語を削除します")
@app_commands.describe(word="削除する単語")
async def delete_dict(interaction: discord.Interaction, word: str):
    if word not in user_dict:
        await interaction.response.send_message(f"`{word}` は辞書に登録されていません。", ephemeral=True)
        return
    del user_dict[word]
    save_dict()
    await interaction.response.send_message(f"🗑️ `{word}` を辞書から削除しました。")


@tree.command(name="テスト読み上げ", description="指定したテキストを即座に読み上げます")
@app_commands.describe(text="読み上げるテキスト")
async def test_read(interaction: discord.Interaction, text: str):
    guild = interaction.guild
    if guild is None or guild.voice_client is None:
        await interaction.response.send_message("先に `/読み上げ開始` を実行してください。", ephemeral=True)
        return
    await interaction.response.send_message(f"🎤 テスト読み上げ: `{text}`")
    if guild.id in read_queues:
        await read_queues[guild.id].put(apply_dict(text))


SCORE_TYPES = ["攻撃換算", "HP換算", "防御換算", "熟知換算"]

SCORE_WEIGHTS = {
    "攻撃換算": {
        "FIGHT_PROP_ATTACK_PERCENT":    1.0,
        "FIGHT_PROP_CHARGE_EFFICIENCY": 0.5,
        "FIGHT_PROP_ELEMENT_MASTERY":   0.25,
    },
    "HP換算": {
        "FIGHT_PROP_HP_PERCENT":        1.0,
        "FIGHT_PROP_CHARGE_EFFICIENCY": 0.5,
        "FIGHT_PROP_ELEMENT_MASTERY":   0.25,
    },
    "防御換算": {
        "FIGHT_PROP_DEFENSE_PERCENT":   1.0,
        "FIGHT_PROP_CHARGE_EFFICIENCY": 0.5,
        "FIGHT_PROP_ELEMENT_MASTERY":   0.25,
    },
    "熟知換算": {
        "FIGHT_PROP_ELEMENT_MASTERY":   1.0,
        "FIGHT_PROP_ATTACK_PERCENT":    0.5,
        "FIGHT_PROP_CHARGE_EFFICIENCY": 0.3,
    },
}

SCORE_BASE = {
    "FIGHT_PROP_ATTACK_PERCENT":    0.049,
    "FIGHT_PROP_HP_PERCENT":        0.049,
    "FIGHT_PROP_DEFENSE_PERCENT":   0.062,
    "FIGHT_PROP_CHARGE_EFFICIENCY": 0.055,
    "FIGHT_PROP_ELEMENT_MASTERY":   19.75,
}

# サブステ1個分の基準値（等価換算の分母）
SCORE_BASE = {
    "FIGHT_PROP_CRITICAL":            0.033,
    "FIGHT_PROP_CRITICAL_HURT":       0.066,
    "FIGHT_PROP_ATTACK_PERCENT":      0.049,
    "FIGHT_PROP_HP_PERCENT":          0.049,
    "FIGHT_PROP_DEFENSE_PERCENT":     0.062,
    "FIGHT_PROP_CHARGE_EFFICIENCY":   0.055,
    "FIGHT_PROP_ELEMENT_MASTERY":     19.75,
}


def calc_artifact_score(artifact, weights: dict) -> float:
    score = 0.0
    sub_stats = getattr(artifact, "sub_stats", [])
    for sub in sub_stats:
        prop_id = str(getattr(sub, "prop_id", ""))
        value   = float(getattr(sub, "value", 0))
        w = weights.get(prop_id, 0)
        base = SCORE_BASE.get(prop_id, 1)
        score += (value / base) * w
    return score





class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def run_health_server():
    port = int(os.getenv("PORT", "8080"))
    HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()


if __name__ == "__main__":
    threading.Thread(target=run_health_server, daemon=True).start()
    bot.run(DISCORD_TOKEN)