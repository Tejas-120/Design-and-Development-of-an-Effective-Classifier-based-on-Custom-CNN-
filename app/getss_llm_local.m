%% getss_llm_local.m
% ---------------------------------------------------------------
% Sends the predicted tumour label + confidence to a locally running
% Ollama server (model: phi3) and returns a plain-language medical
% explanation (description, medical help, symptoms, treatment,
% precautions, when to see a doctor).
%
% Requires Ollama running locally with the 'phi3' model pulled:
%   ollama pull phi3
%   ollama serve   (default endpoint: http://localhost:11434)
%
% NOTE: this function calls fallback_explanation(label, confidence)
% if the LLM call fails, but that helper function was not included
% in the provided code — add a simple fallback_explanation.m (e.g.
% returning a generic templated string) or inline a fallback string
% here before using this in production.
% ---------------------------------------------------------------

function explanation = getss_llm_local(label, confidence)

    try
        disp("CALLING LLM...");

  prompt = "Tumor: " + label + ...
         " (" + num2str(confidence*100,'%.2f') + "%)." + newline + ...
         "Explain in SIMPLE and CLEAR language." + newline + ...
         "Do NOT reduce content. Keep full explanation." + newline + ...
         "Rewrite in easy words that a normal person can understand." + newline + ...
         "Avoid complex medical terms or explain them simply." + newline + ...
         "Format:" + newline + ...
         "Tumor:" + newline + ...
         "Description:" + newline + ...
         "Medical Help:" + newline + ...
         "Symptoms:" + newline + ...
         "Treatment:" + newline + ...
         "Precautions:" + newline + ...
         "Doctor:" + newline + ...
         "When:";

        % ===== API URL =====
        url = "http://localhost:11434/api/generate";

        % ===== REQUEST =====
        data = struct;
        data.model = "phi3";   % You can change to "phi" for faster response
        data.prompt = char(prompt);
        data.stream = false;

        % 🔥 CRITICAL FIX → LIMIT OUTPUT SIZE
        data.options = struct;
        data.options.num_predict = 500;   % 🔥 increase from default

        % ===== OPTIONS =====
        options = weboptions( ...
            'MediaType','application/json', ...
            'Timeout',150);   % Increased safely

        % ===== CALL LLM =====
        response = webwrite(url, data, options);

        disp("LLM RESPONSE:");
        disp(response);

        % ===== SAFE EXTRACTION =====
        if isfield(response,"response") && ~isempty(response.response)
            explanation = response.response;
        else
            error("Empty response from LLM");
        end

    catch ME
        disp("========= LLM FAILED =========");
        disp(ME.message);

        % 🔥 FALLBACK (ALWAYS SAFE)
        explanation = fallback_explanation(label, confidence);
    end

end
