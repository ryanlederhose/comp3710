# Sierpinski Triangle Fractal

"The Sierpiński triangle (sometimes spelled Sierpinski), also called the Sierpiński gasket or Sierpiński sieve, is a fractal attractive fixed set with the overall shape of an equilateral triangle, subdivided recursively into smaller equilateral triangles. Originally constructed as a curve, this is one of the basic examples of self-similar sets—that is, it is a mathematically generated pattern that is reproducible at any magnification or reduction. It is named after the Polish mathematician Wacław Sierpiński, but appeared as a decorative pattern many centuries before the work of Sierpiński." - [Sierpinski Triangle Wiki](https://en.wikipedia.org/wiki/Sierpi%C5%84ski_triangle)

<p align="center">
  <img src="fractal-wiki.png" />
</p>

# Construction
* Pytorch Tensors are used to form a coordinate map of the fractal
* There are 1,000,000 iterations to form the fractal
* Matplotlib used to plot coordinate map of fractal
* GPU accelerated computing
## Iteration Process
1. Vertex of triangle chosen at random
2. Midpoint calaculated between initial point and chosen vertex
3. Midpoint stored into position tensor
4. Initial point is updated to be midpoint
5. Repeat

# Result
<p align="center">
  <img src="Sierpinski-Triangle.png" />
</p>