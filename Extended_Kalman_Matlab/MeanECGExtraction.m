function [ECGmean,ECGsd,meanPhase, K, India, mean_test, std_test, mean_mid] = MeanECGExtraction(x,phase,bins,flag)
%
% [ECGmean,ECGsd,meanPhase] = MeanECGExtraction(x,phase,bins,flag)
% Calculation of the mean and SD of ECG waveforms in different beats
%
% inputs:
% x: input ECG signal
% phase: ECG phase
% bins: number of desired phase bins
% flag
%     1: aligns the baseline on zero, by using the mean of the first 10%
%     segment of the calculated mean ECG beat
%     0: no baseline alignment
%
% outputs:
% ECGmean: mean ECG beat
% ECGsd: standard deviation of ECG beats
% meanPhase: the corresponding phase for one ECG beat
%

meanPhase = zeros(1,bins);
ECGmean = zeros(1,bins);
ECGsd = zeros(1,bins);

I = find( phase>=(pi-pi/bins)  | phase<(-pi+pi/bins) );
India=I
if(~isempty(I))
    meanPhase(1) = -pi;
    mean_test=  mean(x(I))
    ECGmean(1) = mean(x(I));
    std_test=  std(x(I))
    ECGsd(1) = std(x(I));
else
    meanPhase(1) = 0;
    ECGmean(1) =0;
    ECGsd(1) = -1;
end
for i = 1 : bins-1;
    I = find( phase >= 2*pi*(i-0.5)/bins-pi & phase < 2*pi*(i+0.5)/bins-pi );
    if(~isempty(I))
        meanPhase(i + 1) = mean(phase(I));
        ECGmean(i + 1) = mean(x(I));
        ECGsd(i + 1) = std(x(I));
    else
        meanPhase(i + 1) = 0;
        ECGmean(i + 1) = 0;
        ECGsd(i + 1) = -1;
    end
end

mean_mid=ECGmean
K = find(ECGsd==-1);
 for i = 1:length(K)
     switch K(i)
         case 1
             meanPhase(K(i)) = -pi;
             ECGmean(K(i)) = ECGmean(K(i)+1);
             ECGsd(K(i)) = ECGsd(K(i)+1);
         case bins
             meanPhase(K(i)) = pi;
             ECGmean(K(i)) = ECGmean(K(i)-1);
             ECGsd(K(i)) = ECGsd(K(i)-1);
         otherwise
             meanPhase(K(i)) = mean([meanPhase(K(i)-1),meanPhase(K(i)+1)]);
             ECGmean(K(i)) = mean([ECGmean(K(i)-1),ECGmean(K(i)+1)]);
             ECGsd(K(i)) = mean([ECGsd(K(i)-1),ECGsd(K(i)+1)]);
     end
 end

if(flag==1)
    ECGmean = ECGmean - mean(ECGmean(1:ceil(length(ECGmean)/10)));
end