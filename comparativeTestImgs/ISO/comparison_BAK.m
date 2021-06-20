filesHere = dir('*.txt');
in_addresses = {filesHere.name};

PC_Ps = zeros(length(in_addresses),1001);
E_Cos = zeros(length(in_addresses),1001);
envel = zeros(1,1001);

leg = {};

figure(1);
clf;
hold on
plot([0 0.5],[0 0.5], 'r', 'LineWidth', 2, 'HandleVisibility', 'off')
plot([0.5 1],[0.5 0], 'r', 'LineWidth', 2, 'HandleVisibility', 'off')
hold off

for i = 1:length(in_addresses)
   
    disp(in_addresses{i})
    
    fileID = fopen(in_addresses{i},'r');
    
    formatSpec = '%f,%f';
    in_vec_sz = [2 Inf];
    
    in_vec = transpose(fscanf(fileID,formatSpec,in_vec_sz));
    
    PC_P_in = in_vec(:,1);
    E_Co_in = in_vec(:,2);
    
    % increasing resolution
    
    X = 0:0.001:1;
    PC_Ps(i,:) = X;
    
    Y = zeros(1,length(X));
    
    d_X          = -999;
    idx_in_L_old = -999;
    
    for j = 1:(length(X)-1)
       
        idx_in_L = find(PC_P_in > X(j), 1) - 1;
        idx_in_R = idx_in_L + 1;
        
        D_X_in = PC_P_in(idx_in_R) - PC_P_in(idx_in_L);
        D_Y_in = E_Co_in(idx_in_R) - E_Co_in(idx_in_L);
        
        DY_DX_in = D_Y_in / D_X_in;
        
        d_X = d_X + (X(2)-X(1));
        d_X = d_X - (idx_in_L ~= idx_in_L_old)*d_X; % d_X to zero if needed
        
        Y_in_L = E_Co_in(idx_in_L);
        
        Y(j) = Y_in_L + (DY_DX_in * d_X);
        
        idx_in_L_old = idx_in_L;
    end
    
    E_Cos(i,:) = Y;
    
    leg_text = in_addresses{i};
    leg_text = leg_text(5:end-4);
    leg_text = insertBefore(leg_text,'_','\');
    leg{length(leg)+1} = leg_text;
    
    col = hsv2rgb([i/length(in_addresses) 1 0.75]);
    
    figure(1)
    hold on
    
    plot(X,Y, 'Color', col, 'LineWidth',1.2)
    scatter(PC_P_in, E_Co_in,150, col, ...
            '.', 'HandleVisibility', 'off')
    
    hold off
    
end

for i = 1:length(envel)

    envel(i) = min(E_Cos(:,i));
end

leg{length(leg)+1} = 'lower envelope';

figure(1)
hold on

plot(PC_Ps(1,:),envel, '--b', 'LineWidth',3)
legend(leg,'Location','south')

xlim([0 1])
ylim([0 max(max(E_Cos))*1.1])
xlabel('PC(P)')
ylabel('E[Co]')
title('classifier cost curve comparison, ISO scenario')
hold off

set(gcf, 'PaperUnits', 'inches');
x_width = 4;
y_width = 3;
set(gca,'fontsize', 7);
set(gcf, 'PaperPosition', [0 0 x_width y_width]); %
%set(gcf, 'PaperPositionMode', 'auto');
saveas(gcf,  ['./imgs/comparison_ISO.png']);
disp(['saved figure'])


T_out = cat(2,PC_Ps(1,:)',envel');
writematrix(T_out,'./imgs/ISO_low_envel.txt')

