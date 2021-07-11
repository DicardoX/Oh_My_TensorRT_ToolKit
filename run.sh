# !/usr/bin/env bash
# Author: Chunyu Xue


# Model name (default = bert_16)
model_name="bert_16"
# Max batch size (default = 128)
max_batch_size=128
# Type mode (default = 0, is_training)
type_mode=0
# Plot state (default = 0, is_plot)
plot_state=0


# Bert func, [${1}=batch_size, ${2}=type_mode, ${3}=seq_len]
bert_func(){
	if [ ${2} = 0 ];then
		# Training
		# Cp onnx model flie
		if [ ! -e "$(pwd)/custom_python_samples/tensorrt_bert/onnx/bert_${3}.onnx" ];then
			echo ""
			echo "-----------------------------------------------"
			echo "Error:"
			echo "-----------------------------------------------"
			echo "Target onnx file does not exist, please run .py program to generate it..."
			echo "Target .py file path: $(pwd)/custom_python_samples/tensorrt_bert"
			echo "Usage: python bert_to_onnx_dynamic_seq.py --seq_len [SEQ_LEN]"
			echo "-----------------------------------------------"
			echo ""
			exit 1
		fi

		if [ ! -e "$(pwd)/data/bert" ];then
			mkdir $(pwd)/data/resnet
		fi

		cp $(pwd)/custom_python_samples/tensorrt_bert/onnx/bert_${3}.onnx $(pwd)/data/bert/bert_${3}_base_uncased.onnx

		if [ ${1} -gt 32 ];then
			./bin/trtexec --onnx=$(pwd)/data/bert/bert_${3}_base_uncased.onnx --minShapes=input_ids:1x${3},token_type_ids:1x${3},attention_mask:1x${3} --optShapes=input_ids:32x${3},token_type_ids:32x${3},attention_mask:32x${3} --maxShapes=input_ids:${1}x${3},token_type_ids:${1}x${3},attention_mask:${1}x${3} --workspace=4096 --saveEngine=./engines/bert_${3}_dynamic.trt
		else
			./bin/trtexec --onnx=$(pwd)/data/bert/bert_${3}_base_uncased.onnx --minShapes=input_ids:1x${3},token_type_ids:1x${3},attention_mask:1x${3} --optShapes=input_ids:${1}x${3},token_type_ids:${1}x${3},attention_mask:${1}x${3} --maxShapes=input_ids:${1}x${3},token_type_ids:${1}x${3},attention_mask:${1}x${3} --workspace=4096 --saveEngine=./engines/bert_${3}_dynamic.trt
		fi
		exit 1
	elif [ ${2} = 1 ];then
		# Inference

		# -------------------------- Get devices list --------------------------
		str_list=$(nvidia-smi -L)
		device_count=0
		gpu_devices=()

		# Split
		OLD_IFS="$IFS"
		IFS=" "
		# Transfer to array
		str_list=(${str_list})
		IFS="${OLD_IFS}"

		# Classify
		j="${#str_list[@]}"
		for((i=0;i<${j};i++))
		do
			if [ ${str_list[${i}]} = "(UUID:" ];then
				gpu_devices[device_count]=${str_list[${i}+1]//)/}
				device_count=$(($device_count+1))
			fi
		done

		# Final clear
		j="${#gpu_devices[@]}"
		for((i=0;i<${j};i++))
		do
			if [ ${gpu_devices[${i}]:0:1} != "M" ];then
				# Remove all incorrect devices
				unset gpu_devices[i]
			fi
		done
		# ----------------------------------------------------------------------

		# ----------------------------- Check Log Dir --------------------------
		# Check the log dir and recreate it
		if [ -e ./log ];then
			rm -rf ./log
		fi
		mkdir log
		# ----------------------------------------------------------------------

		# ----------------------------- Main Func Loop -------------------------
		for((i=1;i<(${batch_size}+1);i=i*2))
		do
			# # Visit all MIG devices, 1-based in shell
			j="${#gpu_devices[@]}"
			for((k=1;k<${j}+1;k++))
			do
				CUDA_VISIBLE_DEVICES=${gpu_devices[${k}]} ./bin/trtexec --loadEngine=./engines/bert_${3}_dynamic.trt --shapes=input_ids:${i}x${3},token_type_ids:${i}x${3},attention_mask:${i}x${3} > ./log/device_${k}_bs_${i}.txt
				# sleep 30
			done
		done
		# ----------------------------------------------------------------------
	fi
}


# ResNet func, [${1}=batch_size, ${2}=type_mode, ${3}=layer_num]
resnet_func(){
	if [ ${2} = 0 ];then
		# Training
		# Cp onnx model flie
		if [ ! -e "$(pwd)/custom_python_samples/tensorrt_resnet/onnx/resnet_${3}.onnx" ];then
			echo ""
			echo "-----------------------------------------------"
			echo "Error:"
			echo "-----------------------------------------------"
			echo "Target onnx file does not exist, please run .py program to generate it..."
			echo "Target .py file path: $(pwd)/custom_python_samples/tensorrt_resnet"
			echo "Usage: python main.py --layer_num [LAYER_NUM]"
			echo "-----------------------------------------------"
			echo ""
			exit 1
		fi

		if [ ! -e "$(pwd)/data/resnet" ];then
			mkdir $(pwd)/data/resnet
		fi

		cp $(pwd)/custom_python_samples/tensorrt_resnet/onnx/resnet_${3}.onnx $(pwd)/data/resnet/resnet_${3}.onnx

		if [ ${1} -gt 32 ];then
			./bin/trtexec --onnx=$(pwd)/data/resnet/resnet_${3}.onnx --minShapes=input:1x3x224x224 --optShapes=input:32x3x224x224 --maxShapes=input:${1}x3x224x224 --workspace=4096 --saveEngine=./engines/resnet_${3}_dynamic.trt
		else
			./bin/trtexec --onnx=$(pwd)/data/resnet/resnet_${3}.onnx --minShapes=input:1x3x224x224 --optShapes=input:${1}x3x224x224 --maxShapes=input:${1}x3x224x224 --workspace=4096 --saveEngine=./engines/resnet_${3}_dynamic.trt
		fi
		exit 1
	elif [ ${2} = 1 ];then
		# Inference

		# -------------------------- Get devices list --------------------------
		str_list=$(nvidia-smi -L)
		device_count=0
		gpu_devices=()

		# Split
		OLD_IFS="$IFS"
		IFS=" "
		# Transfer to array
		str_list=(${str_list})
		IFS="${OLD_IFS}"

		# Classify
		j="${#str_list[@]}"
		for((i=0;i<${j};i++))
		do
			if [ ${str_list[${i}]} = "(UUID:" ];then
				gpu_devices[device_count]=${str_list[${i}+1]//)/}
				device_count=$(($device_count+1))
			fi
		done

		# Final clear
		j="${#gpu_devices[@]}"
		for((i=0;i<${j};i++))
		do
			if [ ${gpu_devices[${i}]:0:1} != "M" ];then
				# Remove all incorrect devices
				unset gpu_devices[i]
			fi
		done
		# ----------------------------------------------------------------------

		# ----------------------------- Check Log Dir --------------------------
		# Check the log dir and recreate it
		if [ -e ./log ];then
			rm -rf ./log
		fi
		mkdir log
		# ----------------------------------------------------------------------

		# ----------------------------- Main Func Loop -------------------------
		for((i=1;i<(${batch_size}+1);i=i*2))
		do
			# # Visit all MIG devices, 1-based in shell
			j="${#gpu_devices[@]}"
			for((k=1;k<${j}+1;k++))
			do
				CUDA_VISIBLE_DEVICES=${gpu_devices[${k}]} ./bin/trtexec --loadEngine=./engines/resnet_${3}_dynamic.trt --shapes=input:${i}x3x224x224 > ./log/device_${k}_bs_${i}.txt
				# sleep 30
			done
		done
		# ----------------------------------------------------------------------
	fi
}


# Help func
help_message(){
	echo "-----------------------------------------------"
	echo "Usage:"										
	echo "bash ./run.sh [-m MODEL_NAME] [-b MAX_BATCH_SIZE] [-t TYPE_MODE] [-p] [-h]"
	echo "-----------------------------------------------"
	echo "Notice:"
	echo "- NOTICE: run.sh should be put in the main directory of TensorRT (the same dir level as bin and data)."
	echo "-----------------------------------------------"
	echo "Description:"
	echo "- MODEL_NAME: the name of model to be tested (chosen in [bert_16, bert_64, bert_128, resnet_50, resnet_101, resnet_152], default = bert_16)."
	echo "- MAX_BATCH_SIZE: max batch size of the chosen model in inference (pow of 2, default = 128)."
	echo "- TYPE_MODE: 0 / 1 (0 for is_training, 1 for is_inference)"
	echo "- [-p]: Plot result (For inference stage | Path: './output_figs/', 0 for is_plot, 1 for not_plot)"
	echo "- [-h]: help message"
	echo "-----------------------------------------------"
}


# Args
while getopts ":m:b:t:p::h::" opt
do
	case ${opt} in
		m)
		model_name=${OPTARG}
		;;
		b)
		batch_size=${OPTARG}
		;;
		t)
		type_mode=${OPTARG}
		;;
		p)
		plot_state=${OPTARG}
		;;
		h)
		echo ""
		help_message
		echo ""
		;;
		?)
		echo ""
		echo "Invalid argument received..."
		help_message
		echo ""
		exit 1;;
	esac
done

# Command info
echo ""
echo "-----------------------------------------------"
echo "Info:"
echo "- Model name: ${model_name}"
echo "- Max batch size: ${batch_size} (increased from 1 with the pow of 2)"
echo "- Type mode: ${type_mode} (0 for is_training, 1 for is_inference)"
echo "- Plot result ${plot_state} (For inference stage | Path: './output_figs/', 0 for is_plot, 1 for not_plot)"
echo "-----------------------------------------------"
echo ""

# Error detected
if [ ${batch_size} -gt 128 ] || [ ${batch_size} -lt 1 ];then
	echo "Batch size out of range...exit"
	exit 1;
fi

# Function
if [ ${model_name} = "bert_16" ];then
	# Bert_16
	bert_func ${batch_size} ${type_mode} 16
elif [ ${model_name} = "bert_64" ];then
	# Bert_64
	bert_func ${batch_size} ${type_mode} 64
elif [ ${model_name} = "bert_128" ];then
	# Bert_128
	bert_func ${batch_size} ${type_mode} 128
elif [ ${model_name} = "resnet_50" ];then
	# ResNet_50
	resnet_func ${batch_size} ${type_mode} 50
elif [ ${model_name} = "resnet_101" ];then
	# ResNet_101
	resnet_func ${batch_size} ${type_mode} 101
elif [ ${model_name} = "resnet_152" ];then
	# ResNet_152
	resnet_func ${batch_size} ${type_mode} 152
else
	echo ""
	echo "Error:"
	echo "-----------------------------------------------"
	echo "Wrong network choice (optional: [bert_16, bert_64, bert_128, resnet_50, resnet_101, resnet_152])"
	echo "-----------------------------------------------"
	echo ""
	exit 1
fi

# Format output
echo ""
echo ""
echo "#################################### PERFORMANCE SUMMARY ####################################"
python format_output.py

# Plot result
if [ ${plot_state} = 0 ];then
	echo ""
	echo ""
	echo "###################################### PLOTTING RESULT ######################################"
	python curve_plotter.py --model_name ${model_name}
	echo ""
	echo "Plot work is completed! Save pics to path: './output_figs/'"
fi









