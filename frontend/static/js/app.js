document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const inputText = document.getElementById("inputText");
    const modelSelector = document.getElementById("modelSelector");
    const btnAnalyze = document.getElementById("btnAnalyze");
    const btnClear = document.getElementById("btnClear");
    const charCounter = document.getElementById("charCounter");
    const analyzeSpinner = document.getElementById("analyzeSpinner");
    const analyzeIcon = document.getElementById("analyzeIcon");
    const btnText = document.getElementById("btnText");

    // Results Elements
    const resultsPanel = document.getElementById("resultsPanel");
    const verdictBadge = document.getElementById("verdictBadge");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictIconContainer = document.getElementById("verdictIconContainer");
    
    const categoryContainer = document.getElementById("categoryContainer");
    const categoryBadge = document.getElementById("categoryBadge");
    
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceProgressBar = document.getElementById("confidenceProgressBar");
    
    const latencyValue = document.getElementById("latencyValue");
    const languageName = document.getElementById("languageName");
    const highlightedText = document.getElementById("highlightedText");
    const triggerCountBadge = document.getElementById("triggerCountBadge");
    const probBarsContainer = document.getElementById("probBarsContainer");

    const presetButtons = document.querySelectorAll(".preset-pill");

    // Character Counter & Clear button toggler
    inputText.addEventListener("input", () => {
        const len = inputText.value.length;
        charCounter.textContent = `${len} / 500 chars`;
        btnClear.style.display = len > 0 ? "block" : "none";
    });

    btnClear.addEventListener("click", () => {
        inputText.value = "";
        charCounter.textContent = "0 / 500 chars";
        btnClear.style.display = "none";
        resultsPanel.classList.add("d-none");
        inputText.focus();
    });

    // Preset Pill Click Handlers
    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const sample = btn.getAttribute("data-text");
            inputText.value = sample;
            inputText.dispatchEvent(new Event("input"));
            inputText.focus();
            analyzeComment();
        });
    });

    // Keyboard shortcut (Ctrl + Enter) to trigger analysis
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
            alert("Please paste or type a social media comment or tweet to analyze.");
            inputText.focus();
            return;
        }

        const modelChoice = modelSelector ? modelSelector.value : "muril";

        // Set Loading UI State
        setLoadingState(true);

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    text: text,
                    model_choice: modelChoice 
                })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server responded with status ${response.status}`);
            }

            const data = await response.json();
            renderResults(data);

        } catch (err) {
            console.error("Analysis Error:", err);
            alert(`Analysis failed: ${err.message || 'Unable to connect to backend service.'}`);
        } finally {
            setLoadingState(false);
        }
    }

    function setLoadingState(isLoading) {
        btnAnalyze.disabled = isLoading;
        if (isLoading) {
            analyzeSpinner.classList.remove("d-none");
            analyzeIcon.classList.add("d-none");
            btnText.textContent = "Running Neural Inference...";
        } else {
            analyzeSpinner.classList.add("d-none");
            analyzeIcon.classList.remove("d-none");
            btnText.textContent = "Analyze Social Media Text";
        }
    }

    function renderResults(data) {
        // 1. Show Results Panel
        resultsPanel.classList.remove("d-none");

        // 2. Metadata (Latency & Language)
        if (latencyValue) {
            latencyValue.textContent = `${data.latency_ms || 0.0} ms`;
        }
        languageName.textContent = data.language || "English";

        // 3. Verdict Formatting
        const isBullying = data.is_cyberbullying;
        verdictBadge.textContent = data.verdict;

        if (isBullying) {
            verdictBadge.className = "verdict-title m-0 danger";
            verdictIcon.className = "fa-solid fa-triangle-exclamation";
            verdictIconContainer.className = "verdict-icon-container danger";
        } else {
            verdictBadge.className = "verdict-title m-0 safe";
            verdictIcon.className = "fa-solid fa-shield-check";
            verdictIconContainer.className = "verdict-icon-container safe";
        }

        // 4. Category Tag
        if (isBullying && data.category && data.category !== "N/A") {
            categoryContainer.classList.remove("d-none");
            categoryBadge.textContent = data.category;
            categoryBadge.className = `badge category-tag ${data.category}`;
        } else {
            categoryContainer.classList.add("d-none");
        }

        // 5. Confidence Progress Bar Animation
        const conf = data.confidence || 0;
        confidenceValue.textContent = `${conf.toFixed(1)}%`;
        confidenceProgressBar.style.width = `${conf}%`;
        confidenceProgressBar.setAttribute("aria-valuenow", conf);

        if (isBullying) {
            confidenceProgressBar.className = "progress-bar bg-danger-gradient";
        } else {
            confidenceProgressBar.className = "progress-bar bg-success-gradient";
        }

        // 6. Explainability & Highlighted Text
        const exp = data.explainability || {};
        const explainabilityTitle = document.getElementById("explainabilityTitle");
        const explainabilityDesc = document.getElementById("explainabilityDesc");
        const isModelBased = (data.explainability_method && data.explainability_method.includes("Model-Based")) || (data.model_used && data.model_used.includes("MuRIL"));

        if (explainabilityTitle && explainabilityDesc) {
            if (isModelBased) {
                explainabilityTitle.innerHTML = '<i class="fa-solid fa-brain text-primary me-1"></i> Model-Based Token Attribution';
                explainabilityDesc.textContent = 'Gradient-based token attributions derived directly from the fine-tuned Google MuRIL transformer:';
                const tokenCount = (exp.spans && exp.spans.length > 0) ? exp.spans.length : ((exp.top_tokens || []).length);
                triggerCountBadge.textContent = `${tokenCount} token${tokenCount === 1 ? '' : 's'} attributed`;
            } else {
                explainabilityTitle.innerHTML = '<i class="fa-solid fa-highlighter text-warning me-1"></i> Keyword-Based Trigger Detection';
                explainabilityDesc.textContent = 'Dictionary trigger terms and regex patterns contributing to classification are highlighted below:';
                const triggerWords = exp.trigger_words || [];
                triggerCountBadge.textContent = `${triggerWords.length} trigger word${triggerWords.length === 1 ? '' : 's'} flagged`;
            }
        }
        
        if (exp.highlighted_text) {
            highlightedText.innerHTML = exp.highlighted_text;
        } else {
            highlightedText.textContent = data.original_text || inputText.value;
        }

        // 7. Render Probabilities
        if (probBarsContainer && data.all_probabilities) {
            probBarsContainer.innerHTML = "";
            for (const [cls, prob] of Object.entries(data.all_probabilities)) {
                const isTop = (prob === Math.max(...Object.values(data.all_probabilities)));
                const barHtml = `
                    <div class="col-12 col-sm-6">
                        <div class="d-flex justify-content-between fs-8 text-secondary mb-1">
                            <span class="${isTop ? 'fw-700 text-dark' : ''}">${cls}</span>
                            <span class="fw-600">${prob}%</span>
                        </div>
                        <div class="progress" style="height: 6px;">
                            <div class="progress-bar ${isTop ? (isBullying ? 'bg-danger' : 'bg-success') : 'bg-secondary'}" 
                                 style="width: ${prob}%"></div>
                        </div>
                    </div>
                `;
                probBarsContainer.innerHTML += barHtml;
            }
        }

        // Scroll smooth into results view if on small screen
        if (window.innerWidth < 768) {
            resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
});
