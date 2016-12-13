function [max_llrecord, min_llrecord, plots1, plots2] = plot_curve(if_test, records, fig, max1, min1, plots1, plots2, color, line, labelx_ind)
    max_llrecord = max1;
    min_llrecord = min1;
    llrecord_mean = zeros(size(records{1}.llrecord(:,1)));
    errrecord_mean = zeros(size(records{1}.errrecord(:,1)));
    llrecord_mean_test = zeros(size(records{1}.llrecord(:,2)));
    errrecord_mean_test = zeros(size(records{1}.errrecord(:,2)));
    trials = length(records);
    
    alpha = 0.1;
    
    for trial = 1:trials
        iters = size(records{trial}.llrecord, 1) - 1;
        switch labelx_ind
            case 1
                x_inds = 0:iters;
                subplot(1, 2 ,1);
            case 2
                x_inds = records{trial}.eval_fs;
                subplot(2, 1 ,1);
            case 3
                x_inds = records{trial}.eval_gs;
                subplot(2, 1 ,1);
        end
%         trial_curve = semilogy(x_inds, -records{trial}.llrecord(:,1), 'Color', color, 'LineWidth', 1.5);
%         trial_curve.LineStyle = '-';
%         trial_curve.Color(4) = alpha;
%         hold on;
        llrecord_mean = llrecord_mean + (-records{trial}.llrecord(:,1));
        if max(max(-records{trial}.llrecord)) > max_llrecord
            max_llrecord = max(max(-records{trial}.llrecord));
        end
        if min(min(-records{trial}.llrecord)) < min_llrecord
            min_llrecord = min(min(-records{trial}.llrecord));
        end
        if if_test
%             trial_curve_test = semilogy(x_inds, -records{trial}.llrecord(:,2));
%             trial_curve_test.LineStyle = '--';
%             trial_curve_test.Color(4) = 0.2;
%             hold on;
            llrecord_mean_test = llrecord_mean_test + (-records{trial}.llrecord(:,2));
        end
    end
    llrecord_mean = llrecord_mean / trials;
    plot1 = semilogy(x_inds, llrecord_mean, line, 'Color', color, 'LineWidth', 2.5); hold on;
    plots1 = [plots1; plot1];
    if if_test
        llrecord_mean_test = llrecord_mean_test / trials;
        semilogy(x_inds, llrecord_mean_test, '--', 'Color', color, 'LineWidth', 2.5); hold on;
    end
    xlim([0, max(x_inds)]);
    
    for trial = 1:trials
        iters = size(records{trial}.llrecord, 1) - 1;
        switch labelx_ind
            case 1
                x_inds = 0:iters;
                subplot(1, 2 ,2);
            case 2
                x_inds = records{trial}.eval_fs;
                subplot(2, 1 ,2);
            case 3
                x_inds = records{trial}.eval_gs;
                subplot(2, 1 ,2);
        end
%         trial_curve = plot(x_inds, records{trial}.errrecord(:,1), 'Color', color, 'LineWidth', 1.5);
%         trial_curve.LineStyle = '-';
%         trial_curve.Color(4) = alpha;
%         hold on;
        errrecord_mean = errrecord_mean + records{trial}.errrecord(:,1);
        if if_test
%             trial_curve_test = plot(x_inds, records{trial}.errrecord(:,2),'b-'); hold on;
%             trial_curve_test.Color(4) = 0.2;
%             hold on;
            errrecord_mean_test = errrecord_mean_test + (records{trial}.errrecord(:,2));
        end
    end
    errrecord_mean = errrecord_mean / trials;
    plot2 = plot(x_inds, errrecord_mean, line, 'Color', color, 'LineWidth', 2.5); hold on;
    plots2 = [plots2; plot2];
    if if_test
        errrecord_mean_test = errrecord_mean_test / trials;
        plot(x_inds, errrecord_mean_test, '--', 'Color', color, 'LineWidth', 2.5);
    end
    xlim([0, max(x_inds)]);
end