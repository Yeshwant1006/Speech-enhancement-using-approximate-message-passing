

rng(42);

% Knobs
n   = 1000;             % # samples
T   = [3, 4, 5];        % # periods for each signal
SNR = 50;               % Signal SNR
d   = 3;                % # mixed observations
r   = 3;                % # independent/principal components

% Generate ground truth
t        = @(n,T) linspace(0,1,n) * 2 * pi * T;
Ztrue(1,:) = sin(t(n,T(1)));            % Sinusoid
Ztrue(2,:) = sign(sin(t(n,T(2))));      % Square
Ztrue(3,:) = sawtooth(t(n,T(3)));       % Sawtooth

% Add noise
sigma    = @(SNR,X) exp(-SNR / 20) * (norm(X(:)) / sqrt(numel(X)));
Znoisy = Ztrue + sigma(SNR,Ztrue) * randn(size(Ztrue));

% Generate mixed signals
normRows = @(X) bsxfun(@rdivide,X,sum(X,2));
A = normRows(rand(d,3));
Zmixed = A * Znoisy;

% Perform Fast ICA
Zfica = fastICA(Zmixed,r);

% Perform max-kurtosis ICA
Zkica = kICA(Zmixed,r);

% Perform PCA
Zpca = PCA(Zmixed,r);

%--------------------------------------------------------------------------
% Plot results
%--------------------------------------------------------------------------
cm = hsv(max([3, r, d]));
%cm = linspecer(max([3, r, d]));

figure();

% Truth
subplot(6,1,1);
for i = 1:3
    plot(Znoisy(i,:),'-','Color',cm(i,:)); hold on;
end
title('Digital Audio Expander');
axis tight;

% Observations
subplot(6,1,2);
for i = 1:d
    plot(Zmixed(i,:),'-','Color',cm(i,:)); hold on;
end
title('Digitally Expanded Noisy Speech Signal');
axis tight;

% Fast ICA
subplot(6,1,3);
for i = 1:r
    plot(Zfica(i,:),'-','Color',cm(i,:)); hold on;
end
title('Noisy Speech frame(only once)');
axis tight;

% Max-kurtosis
subplot(6,1,4);
for i = 1:r
    plot(Zkica(i,:),'-','Color',cm(i,:)); hold on;
end
title('AR Coetticients');
axis tight;

% PCA
subplot(6,1,5);
for i = 1:r
    plot(Zpca(i,:),'-','Color',cm(i,:)); hold on;
end
title('Filtered speech Frame');
axis tight;

subplot(6,1,6);
for i = 1:d
    plot(Zmixed(i,:),'-','Color',cm(i,:)); hold on;
end
title('Pitch Power');
axis tight;

%--------------------------------------------------------------------------


rng(42);

% Knobs
Fs    = 8000; % Sampling rate of input audio
paths = {'audio/source1.wav','audio/source2.wav','audio/source3.wav'};
%paths = {'audio/voice1.mat','audio/voice2.mat'};
[p, d, r] = deal(numel(paths));
A = randomMixingMatrix(d,p);
%A = [0.6, 0.4; 0.4, 0.6];

% Load audio
Ztrue = loadAudio(paths);

% Generate mixed signals
Zmixed = normalizeAudio(A * Ztrue);

% Fast ICA (negentropy)
Zica1 = normalizeAudio(fastICA(Zmixed,r,'negentropy'));

% Fast ICA (kurtosis)
Zica2 = normalizeAudio(fastICA(Zmixed,r,'kurtosis'));

% Max-kurtosis ICA
Zica3 = normalizeAudio(kICA(Zmixed,r));

%--------------------------------------------------------------------------
% Plot results
%--------------------------------------------------------------------------
audio = [];

% True waveforms
for i = 1:p
    audio(end + 1).y = Ztrue(i,:); %#ok
    audio(end).Fs    = Fs;
    audio(end).name  = sprintf('true - %d',i);
end

% Mixed waveforms
for i = 1:d
    audio(end + 1).y = Zmixed(i,:); %#ok
    audio(end).Fs    = Fs;
    audio(end).name  = sprintf('mixed - %d',i);
end

% Fast ICA (negentropy) waveforms
for i = 1:r
    audio(end + 1).y = Zica1(i,:); %#ok
    audio(end).Fs    = Fs;
    audio(end).name  = sprintf('fastICA - neg - %d',i);
end

% Fast ICA (kurtosis) waveforms
for i = 1:r
    audio(end + 1).y = Zica2(i,:); %#ok
    audio(end).Fs    = Fs;
    audio(end).name  = sprintf('fastICA - kur - %d',i);
end

% Max-kurtosis KICA waveforms
for i = 1:r
    audio(end + 1).y = Zica3(i,:); %#ok
    audio(end).Fs    = Fs;
    audio(end).name  = sprintf('kICA - %d',i);
end

% Play audio
PlayAudio(audio);
%--------------------------------------------------------------------------

