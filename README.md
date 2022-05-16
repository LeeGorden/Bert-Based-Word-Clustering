<br>

<details open>
<summary>Introduction</summary>

A bert-based word clustering algorithm. To cluster clinical comments into different cluster to transfer unconstructed comment data into cluster id for further analytics.

</details>

<details open>
<summary>Step</summary>


- Use one of the 12 pretrained BERT layer to embedding comments.
- Apply t-SNE to lower dimension from 768(Dimension of embedding sentence in BERT) to 2.
- Use GMM(Gaussian Mixture Model) to identify clusters.

</details>

<details open>
<summary>Result</summary>

- [embedding_result.csv](https://github.com/LeeGorden/Bert-Based-Word-Clustering/files/8697458/embedding_result.csv)
  
- ![Word embedding](https://user-images.githubusercontent.com/72702872/168527283-1f91f434-4c78-4391-a63f-1a5d7c9e9e1b.png)

- ![Word embedding_With8clustercolored](https://user-images.githubusercontent.com/72702872/168527291-dd4fc78b-dd90-4107-a687-9e97a8d5f4f1.png)

- ![Word embedding_With8clustercolored_SampleName](https://user-images.githubusercontent.com/72702872/168527315-0abd6256-7d85-4ffa-af94-47eade562bb8.png)

  
</details>![Word embedding_With8clustercolored_SampleName](C:\Users\LiGoudan\Desktop\Git_Upload_TMP\Word Clustering Using Bert\Result\Word embedding_With8clustercolored_SampleName.png)

## <div align="center">Contribute</div>



## <div align="center">Contact</div>

likehao1006@gmail.com

<br>

