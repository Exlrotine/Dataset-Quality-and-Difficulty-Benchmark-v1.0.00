fs = 1e5;                             % 采样频率
% freqRange = [1500 26800];             % 频率范围
freqRange = [26800 50000];             % 频率范围
speedValues = out.Speed(:,2);         % 速度值

startIdx=1;                                        % Start
% startIdx = find(speedValues > 2000, 1, 'first');   % Speedup
% startIdx=880;                                       % Speeddown
% startIdx=3000;                                       % Climbing

signal1 = out.Current(startIdx:(startIdx+6000-1),2); % 信号1（R通道）
signal2 = out.Current(startIdx:(startIdx+6000-1),3); % 信号2（G通道）
signal3 = out.Current(startIdx:(startIdx+6000-1),4); % 信号3（B通道）

[wt1, f] = cwt(signal1, 'amor', fs, 'FrequencyLimits', freqRange, 'VoicesPerOctave', 48);
[wt2, ~] = cwt(signal2, 'amor', fs, 'FrequencyLimits', freqRange, 'VoicesPerOctave', 48);
[wt3, ~] = cwt(signal3, 'amor', fs, 'FrequencyLimits', freqRange, 'VoicesPerOctave', 48);

wt1_abs = abs(wt1);
wt2_abs = abs(wt2);
wt3_abs = abs(wt3);

% 归一化到 [0, 1] 范围，使用三个通道的最大跨度
minVal = min([min(wt1_abs(:)), min(wt2_abs(:)), min(wt3_abs(:))]);
maxVal = max([max(wt1_abs(:)), max(wt2_abs(:)), max(wt3_abs(:))]);
range = maxVal - minVal;

wt1_norm = (wt1_abs - minVal) / range;
wt2_norm = (wt2_abs - minVal) / range;
wt3_norm = (wt3_abs - minVal) / range;

% 创建RGB图像: wt1_norm := R, wt2_norm := G, wt3_norm := B
rgb_image = cat(3, wt1_norm,wt2_norm, wt3_norm);
numTimes = size(rgb_image, 2);                       % 时间轴点数
segmentLength = round(length(f));                    % 每段时间点数
numSegments = ceil(numTimes / segmentLength);        % 总段数

for i = 1:numSegments                                % 直接分段保存图像
    startIdx = (i-1) * segmentLength + 1;
    endIdx = min(i * segmentLength, numTimes);
    rgb_segment = rgb_image(:, startIdx:endIdx, :);  % 提取当前段的RGB数据
    filename = sprintf('006%02d.png', i);            % 格式化文件名
    imwrite(rgb_segment, filename);                  % 直接保存
end

figure('Position', [100 100 800 600]);               % 显示RGB图像
imagesc((0:length(signal1)-1)/fs, f, rgb_image);
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('RGB Composite of Three Signals (CWT)');
ylim(freqRange);