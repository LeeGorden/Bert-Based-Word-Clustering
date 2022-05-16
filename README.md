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


-  [embedding_result.csv](Word Clustering Using Bert\Result\embedding_result.csv) 
- ![Word embedding](C:\Users\LiGoudan\Desktop\Git_Upload_TMP\Word Clustering Using Bert\Result\Word embedding.png)
- <img src="C:\Users\LiGoudan\Desktop\Git_Upload_TMP\Word Clustering Using Bert\Result\Word embedding_With8clustercolored.png" alt="Word embedding_With8clustercolored" style="zoom:72%;" />

</details>![Word embedding_With8clustercolored_SampleName](C:\Users\LiGoudan\Desktop\Git_Upload_TMP\Word Clustering Using Bert\Result\Word embedding_With8clustercolored_SampleName.png)

## <div align="center">Contribute</div>



## <div align="center">Contact</div>

likehao1006@gmail.com

<br>

