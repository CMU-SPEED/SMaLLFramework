# Fusion of Pooling

First there is the question of what the pooling operation is. Typically it is one of the following:
* Max pooling: Max value within the field
* Adaptive pooling: Mean value within the field

Then, looking at the loop nests for convolution and fusion, there are a few different options for fusion and each one has its own\
pros and cons as discussed below.

<table>
<tr>
<th> Convolution </th>
<th> Pooling </th>
</tr>
<tr>
<td>

```c++
for(j = 0; j < K; j++)
{

}
```

</td>
<td>

```c++
int foo() { 
    int x = 4;
    return x;
}
```

</td>
</tr>
</table>