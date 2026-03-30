"""Shared Streamlit UI sections used by both Studio and Photo-to-Animation pages."""

import os
import streamlit as st


def render_video_player(video_path: str, download_name: str = "animation.mp4",
                        dl_key: str = "dl_mp4"):
    """Show a video with a download button."""
    st.markdown("---")
    st.subheader("Generated Animation")
    st.video(video_path)

    with open(video_path, "rb") as f:
        st.download_button(
            label="Download MP4",
            data=f.read(),
            file_name=download_name,
            mime="video/mp4",
            key=dl_key,
        )


def render_teaching_script_section(
    *,
    video_path: str,
    generated_code: str,
    model_choice: str,
    prompt_text: str,
    script_key: str,
    gen_id,
    prefix: str,
    get_video_duration_api,
    generate_teaching_script_api,
):
    """Teaching-script generation + editable text area.

    Returns the (possibly edited) script so the caller can persist it
    back into session state.
    """
    st.markdown("---")
    st.subheader("Teaching Script")

    col_btn, col_info = st.columns([1, 2])

    with col_btn:
        if st.button("Generate Teaching Script", key=f"{prefix}_teach_btn"):
            # fall back to extracting class name when prompt is empty
            effective_prompt = prompt_text
            if not effective_prompt or not effective_prompt.strip():
                import re
                match = re.search(r"class\s+(\w+)", generated_code)
                if match:
                    readable = re.sub(r"([A-Z])", r" \1", match.group(1)).strip()
                    effective_prompt = f"Explain the concept shown in this {readable} animation"

            if not effective_prompt or not effective_prompt.strip():
                st.error("No description available. Please enter a description.")
            else:
                with st.spinner("Creating explanation matched to video..."):
                    dur = get_video_duration_api(video_path)
                    ok, result = generate_teaching_script_api(
                        effective_prompt, generated_code, model_choice, dur
                    )
                    if ok:
                        st.session_state[script_key] = result
                    else:
                        st.error(result)
                st.rerun()

    with col_info:
        dur = get_video_duration_api(video_path)
        if dur > 0:
            st.info(f"Video duration: {dur:.1f}s — script will match this")
        else:
            st.info("The AI will explain the animation in simple words")

    # editable text area
    script = st.session_state.get(script_key)
    if not script:
        return None

    st.markdown("---")
    st.subheader("Teaching Explanation")

    lines = script.count("\n") + 1
    height = min(max(80, lines * 24), 250)

    edited = st.text_area(
        "Teaching explanation:",
        value=script,
        height=height,
        key=f"{prefix}_script_edit_{gen_id}",
    )

    col_save, col_note = st.columns([1, 3])
    with col_save:
        if st.button("Save Script Changes", key=f"{prefix}_save_script_{gen_id}"):
            if edited != script:
                st.session_state[script_key] = edited
                # reset downstream artefacts
                st.session_state[f"{prefix}_teaching_audio"] = None
                st.session_state[f"{prefix}_final_video"] = None
                st.rerun()
    with col_note:
        if edited != script:
            st.caption("Unsaved changes — click Save Script Changes")

    return edited


def render_merge_section(
    *,
    video_path: str,
    script_key: str,
    final_video_key: str,
    final_error_key: str,
    prefix: str,
    merge_video_audio_api,
    code_key: str = "",
):
    """Merge button + final video display + download."""
    script = st.session_state.get(script_key)
    if not script:
        return

    # Pick up generated Manim code for segment-based sync
    manim_code = st.session_state.get(code_key, "") if code_key else ""

    st.markdown("---")
    st.subheader("Create Final Video with Narration")

    col_btn, col_info = st.columns([1, 2])

    with col_btn:
        if st.button("Create Synchronized Video", type="primary", key=f"{prefix}_merge"):
            with st.spinner("Merging animation with narration..."):
                st.session_state[final_video_key] = None
                st.session_state[final_error_key] = None
                ok, result = merge_video_audio_api(video_path, script, manim_code or None)
                if ok:
                    st.session_state[final_video_key] = result
                else:
                    st.session_state[final_error_key] = result
            st.rerun()

    with col_info:
        st.info("Video and audio will be perfectly synchronized")

    # error display
    err = st.session_state.get(final_error_key)
    if err:
        st.error("Video merging failed")
        with st.expander("View Error Details", expanded=True):
            st.code(err, language="text")

    # final video
    final_path = st.session_state.get(final_video_key)
    if final_path and os.path.exists(final_path):
        st.markdown("---")
        st.subheader("Final Video with Audio Narration")
        st.success("Video and audio are now synchronized!")
        st.video(final_path)

        with open(final_path, "rb") as f:
            st.download_button(
                label="Download Final Video",
                data=f.read(),
                file_name=os.path.basename(final_path),
                mime="video/mp4",
                key=f"{prefix}_dl_final",
            )


def render_audio_preview(
    *,
    script_key: str,
    audio_key: str,
    prefix: str,
    generate_tts_api,
):
    """Audio-only preview section."""
    script = st.session_state.get(script_key)
    if not script:
        return

    st.markdown("---")
    st.subheader("Audio Only (Preview)")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Generate Audio Preview", key=f"{prefix}_audio_btn"):
            with st.spinner("Converting to speech..."):
                st.session_state[audio_key] = generate_tts_api(script)
            st.rerun()

    with col2:
        if st.session_state.get(audio_key):
            st.success("Audio ready!")

    audio_path = st.session_state.get(audio_key)
    if audio_path and os.path.exists(audio_path):
        st.audio(audio_path, format="audio/mp3")

        with open(audio_path, "rb") as f:
            st.download_button(
                label="Download Audio Only",
                data=f.read(),
                file_name=f"{prefix}_audio.mp3",
                mime="audio/mp3",
                key=f"{prefix}_dl_audio",
            )
