% dummy data
n = 250;
x = 5*(0:n-1)/n;
y = cos(4*(x - 0.5));
threshold = 0.2*max(y); % 20% of peak amplitude

% find zero crossings
t0_pos1 = find_zc(x, y, threshold);

% plot
figure(1)
plot(x, y, 'b.-', t0_pos1, threshold*ones(size(t0_pos1)), '*r', 'linewidth', 2, 'markersize', 12);
grid on
legend('signal', 'signal positive slope crossing points');

% function to find zero crossings
function [Zx] = find_zc(x, y, threshold)
    y = y - threshold;
    zci = @(data) find(diff(sign(data)) > 0); % function: returns indices of +ZCs
    ix = zci(y); % find indices of + zero crossings of x
    ZeroX = @(x0, y0, x1, y1) x0 - (y0.*(x0 - x1))./(y0 - y1); % Interpolated x value for Zero-Crossing
    Zx = ZeroX(x(ix), y(ix), x(ix+1), y(ix+1));
end
