%% app1.m — Brain Tumour Classifier GUI (MATLAB App Designer)
% ---------------------------------------------------------------
% Simple GUI: upload an MRI image, run it through the trained
% Custom MobileNet multiclass model, and display the predicted
% tumour class + confidence + an LLM-generated plain-language
% explanation (see getss_llm_local.m).
%
% Requires: CustomMobileNet_Singlestage_70_15_15.mat (loaded inside
% the PredicttumorButtonPushed callback) and getss_llm_local.m on
% the MATLAB path.
% ---------------------------------------------------------------

classdef app1 < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure             matlab.ui.Figure
        ExplanationTextArea  matlab.ui.control.TextArea
        ExplanationLabel     matlab.ui.control.Label
        TumorLabel           matlab.ui.control.Label
        ConfidenceLabel      matlab.ui.control.Label
        PredicttumorButton   matlab.ui.control.Button
        UploadImageButton    matlab.ui.control.Button
        UIAxes               matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        imagePath
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: UploadImageButton
        function UploadImageButtonPushed(app, event)

    [file, path] = uigetfile({'*.jpg;*.png;*.jpeg'});
    
    if isequal(file,0)
        return;
    end
    
    app.imagePath = fullfile(path, file);

    img = imread(app.imagePath);
    imshow(img, 'Parent', app.UIAxes);
    axis(app.UIAxes, 'image');

    app.UIAxes.XTick = [];
    app.UIAxes.YTick = [];
    app.UIAxes.Box = 'off';

        end

        % Button pushed function: PredicttumorButton
        function PredicttumorButtonPushed(app, event)


    if isempty(app.imagePath)
        uialert(app.UIFigure, 'Upload image first', 'Error');
        return;
    end

    % Loading message
    app.TumorLabel.Text = "Processing...";
    drawnow;

    % Load model
    load('CustomMobileNet_Singlestage_70_15_15.mat');

    img = imread(app.imagePath);
    img = imresize(img, inputSize(1:2));

    [label, scores] = classify(trainedNet, img);
    confidence = max(scores);

    label = char(label);

    % Clean label text
    label = strrep(label, '_', ' ');
    label(1) = upper(label(1));

    % LLM (local)
    try
        explanation = getss_llm_local(label, confidence);
    catch
        explanation = "Tumor detected: " + label;
    end

    % Update UI
    app.TumorLabel.Text = "Tumor: " + label;
    app.ConfidenceLabel.Text = "Confidence: " + num2str(confidence*100,'%.2f') + "%";
    app.ExplanationTextArea.Value = explanation;
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'MATLAB App';

            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            title(app.UIAxes, 'MRI Image')
            app.UIAxes.Position = [148 209 300 185];

            % Create UploadImageButton
            app.UploadImageButton = uibutton(app.UIFigure, 'push');
            app.UploadImageButton.ButtonPushedFcn = createCallbackFcn(app, @UploadImageButtonPushed, true);
            app.UploadImageButton.Position = [128 434 121 23];
            app.UploadImageButton.Text = 'UploadImageButton';

            % Create PredicttumorButton
            app.PredicttumorButton = uibutton(app.UIFigure, 'push');
            app.PredicttumorButton.ButtonPushedFcn = createCallbackFcn(app, @PredicttumorButtonPushed, true);
            app.PredicttumorButton.Position = [401 434 118 23];
            app.PredicttumorButton.Text = 'PredicttumorButton';

            % Create ConfidenceLabel
            app.ConfidenceLabel = uilabel(app.UIFigure);
            app.ConfidenceLabel.Position = [80 141 88 22];
            app.ConfidenceLabel.Text = 'Confidence: ----';

            % Create TumorLabel
            app.TumorLabel = uilabel(app.UIFigure);
            app.TumorLabel.Position = [90 178 59 22];
            app.TumorLabel.Text = 'Tumor:----';

            % Create ExplanationLabel
            app.ExplanationLabel = uilabel(app.UIFigure);
            app.ExplanationLabel.HorizontalAlignment = 'right';
            app.ExplanationLabel.Position = [108 78 71 22];
            app.ExplanationLabel.Text = 'Explanation:';

            % Create ExplanationTextArea
            app.ExplanationTextArea = uitextarea(app.UIFigure);
            app.ExplanationTextArea.Position = [194 16 412 126];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = app1

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end
