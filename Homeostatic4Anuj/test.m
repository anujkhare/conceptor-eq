% NNets = 8;
% lrr = [0.1 0.05 0.01 0.005 0.001 0.0005];
% no_plots = 1;
% 
% enrgyErrs = zeros(1, length(NNets));
% autoCorrErrs = zeros(1, length(NNets));
% 
% for i = 1:length(lrr)
%    [enrgyErrs(i), autoCorrErrs(i)] = combined_z(NNets, lrr(i), no_plots);
% end
% 
% figure();
% hold on;
% plot(lrr, enrgyErrs, 'rx-');
% plot(lrr, autoCorrErrs, 'gx-');
% hold off;

NNets = 2:2:26;
lrr = 0.01;
no_plots = 1;

enrgyErrs = zeros(1, length(NNets));
autoCorrErrs = zeros(1, length(NNets));

for i = 1:length(NNets)
   [enrgyErrs(i), autoCorrErrs(i)] = combined_z(NNets(i), lrr, no_plots);
end

figure();
hold on;
plot(NNets, enrgyErrs, 'rx-');
plot(NNets, autoCorrErrs, 'gx-');
hold off;