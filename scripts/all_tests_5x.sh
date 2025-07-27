#!/bin/bash

# Each job config line: plot_title|embedding_path|output_name
declare -a job_configs=(
    # || TABLE 4 TESTS || #
    # tests for similar comments
    "Most Similar 5 Comments|verdict_embeddings_postlevel_5.pkl|similar_5_comments"
    "Most Similar 10 Comments|verdict_embeddings_postlevel_10.pkl|similar_10_comments"
    "Most Similar 15 Comments|verdict_embeddings_postlevel_15.pkl|similar_15_comments"
    "Most Similar 20 Comments|verdict_embeddings_postlevel_20.pkl|similar_20_comments"
    "Most Similar 25 Comments|verdict_embeddings_postlevel_25.pkl|similar_25_comments"
    "Most Similar 30 Comments|verdict_embeddings_postlevel_30.pkl|similar_30_comments"

    # tests for similar sentences
    "Most Similar 5 Sentences|verdict_embeddings_sentlevel_5.pkl|similar_5_sentences"
    "Most Similar 10 Sentences|verdict_embeddings_sentlevel_10.pkl|similar_10_sentences"
    "Most Similar 15 Sentences|verdict_embeddings_sentlevel_15.pkl|similar_15_sentences"
    "Most Similar 20 Sentences|verdict_embeddings_sentlevel_20.pkl|similar_20_sentences"
    "Most Similar 25 Sentences|verdict_embeddings_sentlevel_25.pkl|similar_25_sentences"
    "Most Similar 30 Sentences|verdict_embeddings_sentlevel_30.pkl|similar_30_sentences"

    # tests for random comments 
    "5 Random Comments|verdict_embeddings_postlevel_5.pkl|random_5_comments"
    "10 Random Comments|verdict_embeddings_postlevel_10.pkl|random_10_comments"
    "15 Random Comments|verdict_embeddings_postlevel_15.pkl|random_15_comments"
    "20 Random Comments|verdict_embeddings_postlevel_20.pkl|random_20_comments"
    "25 Random Comments|verdict_embeddings_postlevel_25.pkl|random_25_comments"
    "30 Random Comments|verdict_embeddings_postlevel_30.pkl|random_30_comments"

    # tests for random sentences
    "5 Random Sentences|verdict_embeddings_sentlevel_5.pkl|random_5_sentences"
    "10 Random Sentences|verdict_embeddings_sentlevel_10.pkl|random_10_sentences"
    "15 Random Sentences|verdict_embeddings_sentlevel_15.pkl|random_15_sentences"
    "20 Random Sentences|verdict_embeddings_sentlevel_20.pkl|random_20_sentences"
    "25 Random Sentences|verdict_embeddings_sentlevel_25.pkl|random_25_sentences"
    "30 Random Sentences|verdict_embeddings_sentlevel_30.pkl|random_30_sentences"

    # || TABLE 5 TESTS || #

    # tests for theory based clustering
    "Theory Based Clustering - Attitudes|verdict_embeddings_attitudes.pkl|attitudes"
    "Theory Based Clustering - Demographics|verdict_embeddings_demographics.pkl|demographics"
    "Theory Based Clustering - Experiences|verdict_embeddings_experiences.pkl|experiences"
    "Theory Based Clustering - Relationships|verdict_embeddings_relationships.pkl|relationships"


    # tests for automatic clustering
    "Automatic Clustering - Financial Issues|verdict_embeddings_cluster_0.pkl|financial_issues"
    "Automatic Clustering - Acc. Social Needs|verdict_embeddings_cluster_1.pkl|social_needs"
    "Automatic Clustering - Parenting & Discipline|verdict_embeddings_cluster_2.pkl|parenting_discipline"
    "Automatic Clustering - Family Struct. Change|verdict_embeddings_cluster_3.pkl|family_struct_change"
    "Automatic Clustering - Rel. Issues w/Men|verdict_embeddings_cluster_4.pkl|rel_issues_men"
    "Automatic Clustering - Rel. Issues w/Women|verdict_embeddings_cluster_5.pkl|rel_issues_women"
    "Automatic Clustering - Sympathy & Support|verdict_embeddings_cluster_6.pkl|sympathy_support"
    "Automatic Clustering - Family Disputes|verdict_embeddings_cluster_7.pkl|family_disputes"
    "Automatic Clustering - Negative Affect|verdict_embeddings_cluster_8.pkl|negative_affect"
    "Automatic Clustering - Food & Meals|verdict_embeddings_cluster_9.pkl|food_meals"


)

for i in {1..5}; do

    for job in "${job_configs[@]}"; do
        IFS="|" read -r plot_title embedding_path output_name <<< "$job"

        slurm_file="scripts/slurm_auto/${output_name}_${i}.slurm"

        if [[ "$output_name" == random_* ]]; then
            script="ft_bert_no_verdicts.py"
        else
            script="ft_bert_no_verdicts_topk.py"
        fi

        cat > "$slurm_file" <<EOL
#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hendek12@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm/${output_name}_${i}_slurm.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

source env/bin/activate
python src/${script} \
--use_authors='true' \
--author_encoder='average' \
--loss_type='focal' \
--num_epochs=10 \
--sbert_model='sentence-transformers/all-distilroberta-v1' \
--bert_tok='sentence-transformers/all-distilroberta-v1' \
--sbert_dim=768 \
--user_dim=768 \
--model_name='sbert' \
--split_type='sit' \
--situation='text' \
--authors_embedding_path='data/final_embeddings/${embedding_path}' \
--plot_title='${plot_title}_${i}' \
--path_to_data='data/' \
--social_norm='true' \
--log_file='${output_name}_${i}'
EOL

    sbatch "$slurm_file"
    done

done
