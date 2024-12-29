
clear all;

% Neighborhood size parameters
winSize = 5;              % e.g., 5x5 neighborhood
steps = 2;
radius = (winSize - 1)/2;


% Downsample the image to 50x50
M1 = imread('bulbasaur.jpg');
if size(M1, 3) == 3
    M1 = rgb2gray(M1);
end
M1 = double(M1);

[m, n] = size(M1);

% We want to reconstruct to the original 100x100
new_m = m * 2;
new_n = n * 2;

% Initialize the output with padding
output = zeros(new_m + 2*radius, new_n + 2*radius);

% Place the known downsampled pixels
output(radius+1 : 2 : end-radius, radius+1 : 2 : end-radius) = M1;

% Mirror-pad the boundaries
output(1:radius, :) = repmat(output(radius+1, :), radius, 1);
output(end-radius+1:end, :) = repmat(output(end-radius, :), radius, 1);
output(:, 1:radius) = repmat(output(:, radius+1), 1, radius);
output(:, end-radius+1:end) = repmat(output(:, end-radius), 1, radius);

% Setup arrays for accumulation
sum_values = zeros(size(output));
count_values = zeros(size(output));

% Mark known pixels so we never overwrite them
% We will only accumulate and average for pixels that start as zero.
known_mask = (output > 0);  % true where pixels are known initially
tic;
% Interpolation using polynomial fitting, jumping by 2
% Centers at known pixel locations
for i = (radius+1) : steps : (radius + new_m)
    for j = (radius+1) : steps : (radius + new_n)
        % Extract the local neighborhood block
        block = output(i - radius : i + radius, j - radius : j + radius);

        % Find known points in this block
        [x_known, y_known] = find(block > 0);
        z_known = block(block > 0);

        % Shift coordinates to local center
        x_known = x_known - (radius + 1);
        y_known = y_known - (radius + 1);

        if ~isempty(z_known)
            % Construct the A matrix for polynomial fitting
            A = [x_known.^2, x_known.*y_known, y_known.^2, x_known, y_known, ones(size(x_known))];

            % Use pseudoinverse for a least-norm solution
            coeff = pinv(A)*z_known;

            % Compute predictions for every pixel in this block
            for x_local = -radius:radius
                for y_local = -radius:radius
                    global_x = i + x_local;
                    global_y = j + y_local;

                    % We only interpolate unknown pixels (those originally zero)
                    % If this pixel was known from the start, skip to avoid overwriting.
                    if ~known_mask(global_x, global_y)
                        % Predict the pixel value
                        new_value = coeff(1)*x_local^2 + coeff(2)*x_local*y_local + ...
                                    coeff(3)*y_local^2 + coeff(4)*x_local + ...
                                    coeff(5)*y_local + coeff(6);

                        % Accumulate the prediction
                        sum_values(global_x, global_y) = sum_values(global_x, global_y) + new_value;
                        count_values(global_x, global_y) = count_values(global_x, global_y) + 1;
                    end
                end
            end
        end
    end
end
t = toc;
% Compute the averaged output
final_output = output;

% For pixels that were unknown, average them if count_values > 0
avg_mask = (count_values > 0);

% Known pixels remain as is, unknown pixels get averaged values
final_output(avg_mask) = sum_values(avg_mask) ./ count_values(avg_mask);

% Remove the padding
final_output = final_output(radius+1:end-radius, radius+1:end-radius);

% Clamp values to [0, 255]
final_output = max(0, min(255, final_output));

% Display the result
figure;
imagesc(final_output);
colormap(gray);
axis image off;