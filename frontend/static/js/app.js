document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    if (window.lucide) {
        lucide.createIcons();
    }

    // Input Elements
    const inputText = document.getElementById("inputText");
    const btnAnalyze = document.getElementById("btnAnalyze");
    const btnClear = document.getElementById("btnClear");
    const charCounter = document.getElementById("charCounter");
    const analyzeSpinner = document.getElementById("analyzeSpinner");
    const analyzeIconWrapper = document.getElementById("analyzeIconWrapper");
    const btnText = document.getElementById("btnText");

    // Results Elements
    const resultsPanel = document.getElementById("resultsPanel");
    const verdictHeroCard = document.getElementById("verdictHeroCard");
    const verdictBadge = document.getElementById("verdictBadge");
    const verdictSubtitle = document.getElementById("verdictSubtitle");
    const verdictIconContainer = document.getElementById("verdictIconContainer");
    const categoryBadge = document.getElementById("categoryBadge");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceProgressBar = document.getElementById("confidenceProgressBar");
    const languageName = document.getElementById("languageName");
    const highlightedText = document.getElementById("highlightedText");
    const triggerCountBadge = document.getElementById("triggerCountBadge");

    const presetButtons = document.querySelectorAll(".preset-pill");

    // Backend API URL Management
    const RENDER_BACKEND_URL = "https://cyberbullying-detection-8477.onrender.com";

    // Auto-detect backend URL:
    // 1. Check if user set an explicit override in localStorage
    // 2. If running on Hugging Face Spaces (*.hf.space), auto-route to Render backend
    // 3. Otherwise (localhost or on Render domain), use relative path ""
    let backendUrl = localStorage.getItem("guardtext_backend_url");
    if (backendUrl === null) {
        if (window.location.hostname.includes("hf.space")) {
            backendUrl = RENDER_BACKEND_URL;
        } else {
            backendUrl = "";
        }
    }

    const apiEndpointInput = document.getElementById("apiEndpointInput");
    const btnSaveApiEndpoint = document.getElementById("btnSaveApiEndpoint");
    const btnResetApiEndpoint = document.getElementById("btnResetApiEndpoint");

    const errorAlertBanner = document.getElementById("errorAlertBanner");
    const errorMessageText = document.getElementById("errorMessageText");
    const btnCloseAlert = document.getElementById("btnCloseAlert");

    function showError(msg) {
        if (errorMessageText && errorAlertBanner) {
            errorMessageText.textContent = msg;
            errorAlertBanner.classList.remove("d-none");
            errorAlertBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    }

    function hideError() {
        if (errorAlertBanner) {
            errorAlertBanner.classList.add("d-none");
        }
    }

    if (btnCloseAlert) {
        btnCloseAlert.addEventListener("click", hideError);
    }

    // Admin Access trigger (via ?admin=true, #admin, or Ctrl+Shift+B)
    if (window.location.search.includes("admin=true") || window.location.hash === "#admin") {
        const modalEl = document.getElementById("apiModal");
        if (modalEl && window.bootstrap) {
            new bootstrap.Modal(modalEl).show();
        }
    }
    document.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "b") {
            e.preventDefault();
            const modalEl = document.getElementById("apiModal");
            if (modalEl && window.bootstrap) {
                new bootstrap.Modal(modalEl).show();
            }
        }
    });

    if (apiEndpointInput) {
        apiEndpointInput.value = backendUrl || RENDER_BACKEND_URL;
    }

    if (btnSaveApiEndpoint) {
        btnSaveApiEndpoint.addEventListener("click", () => {
            let val = (apiEndpointInput.value || "").trim();
            if (val && !val.startsWith("http://") && !val.startsWith("https://")) {
                val = "https://" + val;
            }
            if (val.endsWith("/")) val = val.slice(0, -1);
            backendUrl = val;
            localStorage.setItem("guardtext_backend_url", backendUrl);
            
            const modalEl = document.getElementById("apiModal");
            if (modalEl && window.bootstrap) {
                const modal = bootstrap.Modal.getInstance(modalEl) || new bootstrap.Modal(modalEl);
                modal.hide();
            }
        });
    }

    if (btnResetApiEndpoint) {
        btnResetApiEndpoint.addEventListener("click", () => {
            backendUrl = "";
            localStorage.removeItem("guardtext_backend_url");
            if (apiEndpointInput) apiEndpointInput.value = "";
        });
    }

    // Character Counter & Clear Button
    inputText.addEventListener("input", () => {
        hideError();
        const len = inputText.value.length;
        charCounter.textContent = `${len} / 500 chars`;
        btnClear.style.display = len > 0 ? "block" : "none";
    });

    btnClear.addEventListener("click", () => {
        hideError();
        inputText.value = "";
        charCounter.textContent = "0 / 500 chars";
        btnClear.style.display = "none";
        resultsPanel.classList.add("d-none");
        inputText.focus();
    });

    // Preset Pills: populate textarea on click WITHOUT auto-analyzing
    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            hideError();
            const sample = btn.getAttribute("data-text");
            inputText.value = sample;
            inputText.dispatchEvent(new Event("input"));
            inputText.focus();
            // Hide previous results panel until user clicks the Analyze button
            resultsPanel.classList.add("d-none");
        });
    });

    // Keyboard Shortcut: Ctrl + Enter
    inputText.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            analyzeComment();
        }
    });

    btnAnalyze.addEventListener("click", analyzeComment);

    async function analyzeComment() {
        const text = inputText.value.trim();
        if (!text) {
            inputText.focus();
            return;
        }

        hideError();
        setLoadingState(true);

        try {
            // STRICTLY query the fine-tuned Google MuRIL Transformer model
            const targetUrl = backendUrl ? `${backendUrl}/api/analyze` : "/api/analyze";
            const response = await fetch(targetUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    text: text, 
                    model_choice: "muril" 
                })
            });

            if (!response.ok) {
                if (response.status === 404 && !backendUrl) {
                    const modalEl = document.getElementById("apiModal");
                    if (modalEl && window.bootstrap) {
                        const modal = new bootstrap.Modal(modalEl);
                        modal.show();
                    }
                    throw new Error("Static Hugging Face Space requires a connection to your Python model server. Enter backend URL in settings.");
                }
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server responded with status ${response.status}`);
            }

            const data = await response.json();
            renderResults(data, text);

        } catch (err) {
            console.error("Model Inference Error:", err);
            showError("Analysis Notice: " + (err.message || "Failed to reach model server."));
        } finally {
            setLoadingState(false);
        }
    }

    function setLoadingState(isLoading) {
        btnAnalyze.disabled = isLoading;
        if (isLoading) {
            analyzeSpinner.classList.remove("d-none");
            if (analyzeIconWrapper) analyzeIconWrapper.classList.add("d-none");
            btnText.textContent = "Running MuRIL Model...";
        } else {
            analyzeSpinner.classList.add("d-none");
            if (analyzeIconWrapper) analyzeIconWrapper.classList.remove("d-none");
            btnText.textContent = "Analyze Comment";
        }
    }

    function renderResults(data, originalText) {
        resultsPanel.classList.remove("d-none");

        // 1. Detected Language Display (Non-interactive metadata label)
        if (languageName) {
            languageName.textContent = data.language || "English";
        }

        const isBully = Boolean(data.is_cyberbullying);
        const rawCategory = data.category || (isBully ? "Other" : "Safe");
        const category = rawCategory.toLowerCase() === "not_cyberbullying" ? "Safe" : rawCategory;
        const confidence = parseFloat(data.confidence) || 0.0;

        // 2. Primary Hero Element: Detection Verdict Card
        if (isBully) {
            verdictBadge.textContent = "Cyberbullying Detected";
            verdictBadge.className = "verdict-title m-0 danger";
            if (verdictSubtitle) {
                verdictSubtitle.textContent = "Harmful or abusive language flagged by model";
            }
            if (verdictHeroCard) {
                verdictHeroCard.className = "verdict-hero-card mb-4 danger";
            }
            if (verdictIconContainer) {
                verdictIconContainer.className = "verdict-icon-box danger";
                verdictIconContainer.innerHTML = '<i data-lucide="alert-triangle" class="icon-hero"></i>';
            }

            // Target Category: Secondary muted purple / neutral tone (does not compete with red alert)
            categoryBadge.textContent = category.toUpperCase();
            categoryBadge.className = "category-pill-badge active-category";

        } else {
            verdictBadge.textContent = "Safe Content";
            verdictBadge.className = "verdict-title m-0 safe";
            if (verdictSubtitle) {
                verdictSubtitle.textContent = "No harmful or targeted harassment detected";
            }
            if (verdictHeroCard) {
                verdictHeroCard.className = "verdict-hero-card mb-4 safe";
            }
            if (verdictIconContainer) {
                verdictIconContainer.className = "verdict-icon-box safe";
                verdictIconContainer.innerHTML = '<i data-lucide="shield-check" class="icon-hero"></i>';
            }

            // Target Category: Neutral muted gray badge for Safe
            categoryBadge.textContent = "NONE / N/A";
            categoryBadge.className = "category-pill-badge Safe";
        }

        // 3. Dynamic Model Confidence Bar (Amber 50-70%, Red 70%+, Green Safe)
        let severityClass = "safe";
        if (isBully) {
            if (confidence >= 70.0) {
                severityClass = "severity-high"; // Alert Red (70%+)
            } else {
                severityClass = "severity-medium"; // Warning Amber (50% - 70%)
            }
        } else {
            severityClass = "safe"; // Success Green
        }

        confidenceProgressBar.className = `progress-bar custom-progress-fill ${severityClass}`;
        confidenceValue.className = `confidence-val ${severityClass}`;
        confidenceValue.textContent = `${confidence.toFixed(1)}%`;
        confidenceProgressBar.style.width = `${Math.min(100, Math.max(0, confidence))}%`;

        // 4. Keyword-Based Trigger Detection
        const explain = data.explainability || {};
        const triggerWords = explain.trigger_words || [];
        const spans = explain.spans || [];
        const count = triggerWords.length > 0 ? triggerWords.length : spans.length;

        const triggerMethodText = document.getElementById("triggerMethodText");
        if (triggerMethodText) {
            triggerMethodText.textContent = data.explainability_method || "Keyword-Based Trigger Detection";
        }

        triggerCountBadge.textContent = count === 1 ? "1 trigger word flagged" : `${count} trigger words flagged`;

        if (explain.highlighted_text && explain.highlighted_text.trim()) {
            highlightedText.innerHTML = explain.highlighted_text;
        } else {
            highlightedText.textContent = originalText;
        }

        // Re-hydrate dynamic Lucide icons in newly rendered HTML
        if (window.lucide) {
            lucide.createIcons();
        }

        // Smooth scroll to results
        resultsPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
});
