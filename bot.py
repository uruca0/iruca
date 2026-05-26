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
import numpy as np
import pyopenjtalk
from discord import app_commands
from discord.ext import commands

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "MTEwNzY1MTg2MzY0NzYzNzUxNA.GhiVry.bOr1IiqS_MaF4IGLPPxncntNkCl70Wu-X_e5l8")
DICT_FILE = Path("openjtalk_dict.json")
IGNORE_PREFIXES = ("!", "/", ".", "?")
SAMPLE_RATE = 48000

active_channels: dict = {}
read_queues: dict = {}
user_dict: dict = {}


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

        # float64 [-1.0, 1.0] に正規化
        peak = np.max(np.abs(wave_data))
        if peak > 0:
            wave_data = wave_data / peak

        # ソフトリミッター（閾値0.8でなめらかに圧縮）
        wave_data = soft_limit(wave_data, threshold=0.8)

        # 音量を0.85倍に抑えてヘッドルームを確保
        wave_data = wave_data * 0.85

        # float → int16
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


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)