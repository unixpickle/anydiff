package anyfwd

import "github.com/unixpickle/anyvec"

// AddChunks adds a different scalar to each chunk in v.
func (v *Vector) AddChunks(scalars anyvec.Vector) {
	v.additiveChunkOp(scalars, anyvec.AddChunks)
}

// ScaleChunks scales chunks of v by different scalers.
func (v *Vector) ScaleChunks(scalers anyvec.Vector) {
	v.multiplicativeChunkOp(scalers, anyvec.ScaleChunks)
}

// AddRepeated adds a repeating form of v1 to v.
func (v *Vector) AddRepeated(v1 anyvec.Vector) {
	v.additiveChunkOp(v1, anyvec.AddRepeated)
}

// ScaleRepeated multiplies v component-wise by a repeated
// form of scalers.
func (v *Vector) ScaleRepeated(scalers anyvec.Vector) {
	v.multiplicativeChunkOp(scalers, anyvec.ScaleRepeated)
}

func (v *Vector) additiveChunkOp(v1 anyvec.Vector, f func(v1, v2 anyvec.Vector)) {
	vec := v.convertVec(v1)
	f(v.Values, vec.Values)
	for i, grad := range v.Jacobian {
		f(grad, vec.Jacobian[i])
	}
}

func (v *Vector) multiplicativeChunkOp(v1 anyvec.Vector, f func(v1, v2 anyvec.Vector)) {
	vec := v.convertVec(v1)
	oldVal := v.Values.Copy()
	f(v.Values, vec.Values)
	for i, grad := range v.Jacobian {
		// Product rule.
		product1 := oldVal.Copy()
		f(product1, vec.Jacobian[i])
		f(grad, vec.Values)
		grad.Add(product1)
	}
}
