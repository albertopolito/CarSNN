%-------------------------------------------------------------------------%
% This script can be used                                                 %
% a)to change the format of images in order to have the input data for the%
%   python script                                                         %
% b)to visualize the occurences of events in the xy coordinate space      %
%-------------------------------------------------------------------------%

write_output_files_test=0; %set to 1 if you would write the new dataset 
                           %files into the format used in python script 
                            

write_output_files_train=0; %set to 1 if you would write the new dataset 
                            %files into the format used in python script

% variables to store the occurences of events                           
occurences_train=zeros(304,304);
occurences_test=zeros(304,304);

%% test part

% create new directories to store the files 
if write_output_files_test==1
    mkdir N_cars
    mkdir ./N_cars test
    mkdir ./N_cars/test cars
    mkdir ./N_cars/test background
end

% cars images
for i=0:4395
    filename_from=sprintf('./n-cars_test/test/cars/obj_%06d_td.dat',i);
    if write_output_files_test==1
        filename_to=sprintf('./N_cars/test/cars/obj_%06d_td.dat',i);
        to= fopen(filename_to,'w');
    end
    data=load_atis_data(filename_from);
    for j=1:length(data.ts)
        % increase the matrix of occurrences
        occurences_test(data.x(j)+1,data.y(j)+1)=1;
        if write_output_files_test==1
            % write the output file in format: 
            % timestamp x-coordinate y-coordinate polarity 
            fprintf(to, '%d %d %d %d\n',data.ts(j),data.x(j),data.y(j),data.p(j));
        end
    end
    if write_output_files_test==1
        fclose(to);
    end
end

% background images
for i=0:4210
    filename_from=sprintf('./n-cars_test/test/background/obj_%06d_td.dat',i);
    if write_output_files_test==1
        filename_to=sprintf('./N_cars/test/background/obj_%06d_td.dat',i);
        to= fopen(filename_to,'w');
    end
    data=load_atis_data(filename_from);
    for j=1:length(data.ts)
        % increase the matrix of occurrences
        occurences_test(data.x(j)+1,data.y(j)+1)=1;
        if write_output_files_test==1
            % write the output file in format: 
            % timestamp x-coordinate y-coordinate polarity
            fprintf(to, '%d %d %d %d\n',data.ts(j),data.x(j),data.y(j),data.p(j));
        end
    end
    if write_output_files_test==1
        fclose(to);
    end
end

%% train part

% create new directories to store the files
if write_output_files_train==1
    mkdir N_cars

    mkdir ./N_cars train
    mkdir ./N_cars/train cars
    mkdir ./N_cars/train background
end

%cars images
for i=4396:12335
    filename_from=sprintf('./n-cars_train/train/cars/obj_%06d_td.dat',i);
    if write_output_files_train==1
        filename_to=sprintf('./N_cars/train/cars/obj_%06d_td.dat',i);
        to= fopen(filename_to,'w');
    end
    data=load_atis_data(filename_from);
    for j=1:length(data.ts)
        % increase the matrix of occurrences
        occurences_train(data.x(j)+1,data.y(j)+1)=1;
        if write_output_files_train==1
            % write the output file in format: 
            % timestamp x-coordinate y-coordinate polarity
            fprintf(to, '%d %d %d %d\n',data.ts(j),data.x(j),data.y(j),data.p(j));
        end
    end
    if write_output_files_train==1
        fclose(to);
    end
end

% background images
for i=4211:11692
    filename_from=sprintf('./n-cars_train/train/background/obj_%06d_td.dat',i);
    if write_output_files_train==1
        filename_to=sprintf('./N_cars/train/background/obj_%06d_td.dat',i);
        to= fopen(filename_to,'w');
    end
    data=load_atis_data(filename_from);
    for j=1:length(data.ts)
        % increase the matrix of occurrences
        occurences_train(data.x(j)+1,data.y(j)+1)=1;
        if write_output_files_train==1
            % write the output file in format: 
            % timestamp x-coordinate y-coordinate polarity
            fprintf(to, '%d %d %d %d\n',data.ts(j),data.x(j),data.y(j),data.p(j));
        end
    end
    if write_output_files_train==1
        fclose(from);
    end
end

%% occurences graphs
figure(1)
pcolor(occurences_train)
colorbar
title('Occurrences of train')
xlabel('x')
ylabel('y')

figure(2)
pcolor(occurences_test)
colorbar
title('Occurrences of test')
xlabel('x')
ylabel('y')