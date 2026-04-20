from policy_compliance_agent.core.transcripts import extract_speaker_text, load_transcripts_from_folder, load_transcripts_structured_from_txt


def test_load_plain_and_diarized_transcripts(tmp_path):
    diarized = tmp_path / "call1.txt"
    diarized.write_text(
        "SPEAKER_00 [1.00s - 2.00s]: Good morning mister tan\n"
        "SPEAKER_01 [2.00s - 3.00s]: Yes speaking\n"
        "SPEAKER_00 [3.00s - 4.00s]: there is a risk mismatch for this trade\n",
        encoding="utf-8",
    )
    plain = tmp_path / "call2.txt"
    plain.write_text("This is a plain transcript without diarization.", encoding="utf-8")

    transcripts = load_transcripts_from_folder(tmp_path)
    structured = load_transcripts_structured_from_txt(tmp_path)

    assert {item["transcript_id"] for item in transcripts} == {"call1", "call2"}
    diarized_text = next(item["transcript"] for item in transcripts if item["transcript_id"] == "call1")
    assert "risk mismatch" in diarized_text

    structured_call = next(item for item in structured if item["transcript_id"] == "call1")
    assert structured_call["transcript"]["sca"][0].startswith("Good morning")
    assert structured_call["transcript"]["client"][0] == "Yes speaking"


def test_loaded_colon_diarized_transcript_keeps_agent_turn_boundaries(tmp_path):
    transcript_path = tmp_path / "travel_call.txt"
    transcript_path.write_text(
        "Agent: There will be a fee to make the change.\n"
        "Customer: That sounds fine.\n"
        "Agent: I can confirm once the pricing view settles.\n",
        encoding="utf-8",
    )

    transcript = load_transcripts_from_folder(tmp_path)[0]["transcript"]
    agent_only = extract_speaker_text(transcript, ["Agent"])

    assert "There will be a fee" in agent_only
    assert "I can confirm" in agent_only
    assert "Customer:" not in agent_only
    assert "That sounds fine" not in agent_only
