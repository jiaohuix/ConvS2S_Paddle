python generate.py --cfg configs/en2ro.yaml \
				   --test-pref wmt16_enro_bpe/test \
				   --only-src \
				   --pretrained ./model_best \
				   --beam-size 5 \
#				   --generate-path generate.txt \
#				   --sorted-path result.txt
