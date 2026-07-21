let currentMode = "rag";

// ── MODE SELECTION ─────────────────────────────────────
function setMode(mode, btn) {
    currentMode = mode;
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
}

// ── QUICK SEND ─────────────────────────────────────────
function sendQuick(text) {
    document.getElementById("user-input").value = text;
    sendMessage();
}

// ── CLEAR CHAT ─────────────────────────────────────────
function clearChat() {
    const messages = document.getElementById("messages");
    messages.innerHTML = `
        <div class="message bot-message">
            <div class="avatar bot-avatar">🤖</div>
            <div class="bubble bot-bubble">
                <p>Chat cleared! Ask me anything about placement records from 2023–2025.</p>
                <div class="bubble-time">Just now</div>
            </div>
        </div>`;
}

// ── TIME HELPER ────────────────────────────────────────
function getTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ── FORMAT BOT ANSWER ──────────────────────────────────
function stripBoldMarkers(text) {
    return String(text || "").replace(/\*\*/g, "").trim();
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    }[char]));
}

function isMarkdownTable(lines) {
    return lines.length >= 2 && lines[0].includes("|") && /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[1]);
}

function isTableSeparator(line) {
    return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function isMarkdownTableAt(lines, index) {
    return Boolean(lines[index] && lines[index + 1] && lines[index].includes("|") && isTableSeparator(lines[index + 1]));
}

function renderMarkdownTable(lines) {
    const rows = lines
        .filter((line, index) => index !== 1 && line.includes("|"))
        .map(line => line.trim().replace(/^\||\|$/g, "").split("|").map(cell => escapeHtml(stripBoldMarkers(cell.trim()))));

    if (!rows.length) return "";

    const headers = rows[0].map(cell => `<th>${cell}</th>`).join("");
    const body = rows.slice(1).map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join("")}</tr>`).join("");
    return `<div class="bot-table-wrap"><table class="bot-table"><thead><tr>${headers}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function isListLine(line) {
    return /^([-*]|\d+\.)\s+/.test(line);
}

function renderListBlock(lines) {
    const listTag = lines.every(l => /^\d+\.\s+/.test(l)) ? "ol" : "ul";
    const items = lines
        .map(l => `<li>${escapeHtml(l.replace(/^([-*]|\d+\.)\s+/, ""))}</li>`)
        .join("");
    return `<${listTag} class="bot-list">${items}</${listTag}>`;
}

function flushParagraph(buffer, html) {
    if (!buffer.length) return;
    html.push(`<p>${buffer.map(escapeHtml).join("<br>")}</p>`);
    buffer.length = 0;
}

function formatAnswer(text) {
    const lines = stripBoldMarkers(text).split("\n").map(l => l.trim()).filter(Boolean);
    if (!lines.length) return "";

    if (lines.some((line, index) => isMarkdownTableAt(lines, index) || isListLine(line))) {
        const html = [];
        const paragraph = [];

        for (let i = 0; i < lines.length; i++) {
            if (isMarkdownTableAt(lines, i)) {
                flushParagraph(paragraph, html);
                const tableLines = [lines[i], lines[i + 1]];
                i += 2;
                while (i < lines.length && lines[i].includes("|")) {
                    tableLines.push(lines[i]);
                    i++;
                }
                i--;
                html.push(renderMarkdownTable(tableLines));
                continue;
            }

            if (isListLine(lines[i])) {
                flushParagraph(paragraph, html);
                const listLines = [];
                while (i < lines.length && isListLine(lines[i])) {
                    listLines.push(lines[i]);
                    i++;
                }
                i--;
                html.push(renderListBlock(listLines));
                continue;
            }

            paragraph.push(lines[i]);
        }

        flushParagraph(paragraph, html);
        return html.join("");
    }

    if (isMarkdownTable(lines)) {
        return renderMarkdownTable(lines);
    }

    const bulletLines = lines.filter(l => /^([-*]|\d+\.)\s+/.test(l));
    if (bulletLines.length >= 2) {
        const intro = lines.filter(l => !/^([-*]|\d+\.)\s+/.test(l)).map(escapeHtml).join("<br>");
        const listTag = bulletLines.every(l => /^\d+\.\s+/.test(l)) ? "ol" : "ul";
        const items = bulletLines
            .map(l => `<li>${escapeHtml(l.replace(/^([-*]|\d+\.)\s+/, ""))}</li>`)
            .join("");
        return (intro ? `<p>${intro}</p>` : "") + `<${listTag} class="bot-list">${items}</${listTag}>`;
    }

    const isNumberedList = lines.filter(l => /^\d+\.\s/.test(l.trim())).length > 3;

    if (isNumberedList) {
        const intro = lines.filter(l => !/^\d+\.\s/.test(l.trim())).map(escapeHtml).join("<br>");
        const items = lines
            .filter(l => /^\d+\.\s/.test(l.trim()))
            .map(l => `<div class="company-item">${escapeHtml(l.trim())}</div>`)
            .join("");
        return `<p>${intro}</p><div class="company-list">${items}</div>`;
    }

    // Normal text — replace newlines with <br>
    return "<p>" + lines.map(escapeHtml).join("</p><p>") + "</p>";
}

// ── MAIN SEND ──────────────────────────────────────────
async function sendMessage() {
    const input   = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const messages = document.getElementById("messages");
    const typing  = document.getElementById("typing");

    const question = input.value.trim();
    if (!question) return;

    // Add user bubble
    messages.innerHTML += `
        <div class="message user-message">
            <div class="avatar user-avatar">U</div>
            <div class="bubble user-bubble">
                ${question}
                <div class="bubble-time">${getTime()}</div>
            </div>
        </div>`;

    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;
    typing.style.display = "flex";
    messages.scrollTop = messages.scrollHeight;

    try {
        const response = await fetch("http://127.0.0.1:8000/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, mode: currentMode })
        });

        const data = await response.json();
        const formatted = formatAnswer(data.answer || "No response received.");

        messages.innerHTML += `
            <div class="message bot-message">
                <div class="avatar bot-avatar">🤖</div>
                <div class="bubble bot-bubble">
                    ${formatted}
                    <div class="bubble-time">${getTime()} · ${currentMode === "rag" ? "RAG" : "Zero Shot"}</div>
                </div>
            </div>`;

    } catch (err) {
        messages.innerHTML += `
            <div class="message bot-message">
                <div class="avatar bot-avatar">🤖</div>
                <div class="bubble bot-bubble" style="border-color:#ef4444;">
                    <p>⚠️ Could not connect to backend. Make sure <code>app.py</code> is running.</p>
                    <div class="bubble-time">${getTime()}</div>
                </div>
            </div>`;
    }

    typing.style.display = "none";
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
    messages.scrollTop = messages.scrollHeight;
}

// ── ENTER KEY ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("user-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});

async function handleResumeUpload() {
    const fileInput = document.getElementById('resumeFile');
    const statusDiv = document.getElementById('resumeStatus');
    const messagesContainer = document.getElementById('messages');
    
    if (!fileInput.files || fileInput.files.length === 0) return;
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading text
    statusDiv.style.display = "block";
    statusDiv.innerText = "Analyzing resume keywords...";
    
    try {
        const response = await fetch("http://127.0.0.1:8000/api/upload-resume", {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        statusDiv.style.display = "none";
        
        if (data.success && data.suggestions.length > 0) {
            // Build a clean message list displaying matching results
            let resultsHtml = `<p>📊 <strong>Resume Analysis Complete!</strong></p>`;
            resultsHtml += `<p>Based on your profile skills and past recruitment records (2023-2025), here are the top companies you are highly suitable for:</p><ol style="margin-top: 8px; padding-left: 16px;">`;
            
            data.suggestions.forEach(item => {
                resultsHtml += `<li style="margin-bottom: 4px;"><strong>${item.company}</strong></li>`;
            });
            resultsHtml += `</ol><p style="margin-top: 8px; font-size: 13px; color: #8e95b0;">Try asking PlaceBot about deadlines or interview requirements for these corporations!</p>`;
            
            // Push the result into the chat container
            messagesContainer.innerHTML += `
                <div class="message bot-message">
                    <div class="avatar bot-avatar">🤖</div>
                    <div class="bubble bot-bubble">
                        ${resultsHtml}
                        <div class="bubble-time">Just now · Profile Match</div>
                    </div>
                </div>`;
        } else {
            const msg = data.message || "We couldn't find explicit historical overlap matches for keywords in this specific document configuration.";
            messagesContainer.innerHTML += `
                <div class="message bot-message">
                    <div class="avatar bot-avatar">🤖</div>
                    <div class="bubble bot-bubble" style="border-color:#ef4444;">
                        <p>⚠️ ${msg}</p>
                    </div>
                </div>`;
        }
    } catch (err) {
        statusDiv.style.display = "none";
        alert("Could not connect to the backend server. Please make sure app.py is actively running.");
    }
    
    // Clear the input so you can re-upload if needed
    fileInput.value = "";
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
