package anyfwd

import "github.com/unixpickle/anyvec"

// Vector is a dual vector.
//
// The fields are setup similarly to those in NumericList.
type Vector struct {
	CreatorPtr *Creator
	Values     anyvec.Vector
	Jacobian   []anyvec.Vector
}

// Creator returns v.CreatorPtr.
func (v *Vector) Creator() anyvec.Creator {
	return v.CreatorPtr
}

// Len returns the vector length.
func (v *Vector) Len() int {
	return v.Values.Len()
}

// Overlaps checks if the vectors overlap.
func (v *Vector) Overlaps(v1 anyvec.Vector) bool {
	return v.Values.Overlaps(v1.(*Vector).Values)
}

// Data generates a NumericList for the vector.
func (v *Vector) Data() anyvec.NumericList {
	res := NumericList{Values: v.Values.Data()}
	for _, x := range v.Jacobian {
		res.Jacobian = append(res.Jacobian, x.Data())
	}
	return res
}

// SetData updates the vector's values from a NumericList.
func (v *Vector) SetData(data anyvec.NumericList) {
	nl := data.(NumericList)
	if len(nl.Jacobian) != len(v.Jacobian) {
		panic(badJacobianErr)
	}
	v.Values.SetData(nl.Values)
	for i, x := range nl.Jacobian {
		v.Jacobian[i].SetData(x)
	}
}

// Set copies v1 into v.
func (v *Vector) Set(v1 anyvec.Vector) {
	vec1 := v1.(*Vector)
	if len(v.Jacobian) != len(vec1.Jacobian) {
		panic(badJacobianErr)
	}
	v.Values.Set(vec1.Values)
	for i, x := range vec1.Jacobian {
		v.Jacobian[i].Set(x)
	}
}

// Copy copies the vector.
func (v *Vector) Copy() anyvec.Vector {
	v1 := &Vector{CreatorPtr: v.CreatorPtr, Values: v.Values.Copy()}
	for _, x := range v.Jacobian {
		v1.Jacobian = append(v1.Jacobian, x.Copy())
	}
	return v1
}

// Slice creates an alias to a sub-range of the vector.
func (v *Vector) Slice(start, end int) anyvec.Vector {
	v1 := &Vector{CreatorPtr: v.CreatorPtr, Values: v.Values.Slice(start, end)}
	for _, x := range v.Jacobian {
		v1.Jacobian = append(v1.Jacobian, x.Slice(start, end))
	}
	return v1
}

// Scale scales the vector by a Numeric.
func (v *Vector) Scale(s anyvec.Numeric) {
	num := v.convertNum(s)
	for i, grad := range v.Jacobian {
		// Product rule.
		grad.Scale(num.Value)
		scaledVal := v.Values.Copy()
		scaledVal.Scale(num.Grad[i])
		grad.Add(scaledVal)
	}
	v.Values.Scale(num.Value)
}

// AddScalar adds a Numeric to the vector.
func (v *Vector) AddScalar(s anyvec.Numeric) {
	num := v.convertNum(s)
	v.Values.AddScalar(num.Value)
	for i, x := range v.Jacobian {
		x.AddScalar(num.Grad[i])
	}
}

// Dot computes a dot product.
func (v *Vector) Dot(v1 anyvec.Vector) anyvec.Numeric {
	vec1 := v.convertVec(v1)
	res := Numeric{Value: v.Values.Dot(vec1.Values)}
	for i, grad := range v.Jacobian {
		grad1 := vec1.Jacobian[i]
		// Product rule.
		n1 := grad1.Dot(v.Values)
		n2 := grad.Dot(vec1.Values)
		res.Grad = append(res.Grad, v.addNumerics(n1, n2))
	}
	return res
}

// Add performs component-wise addition.
func (v *Vector) Add(v1 anyvec.Vector) {
	vec1 := v.convertVec(v1)
	v.Values.Add(vec1.Values)
	for i, grad := range v.Jacobian {
		grad.Add(vec1.Jacobian[i])
	}
}

// Sub performs component-wise subtraction.
func (v *Vector) Sub(v1 anyvec.Vector) {
	vec1 := v.convertVec(v1)
	v.Values.Sub(vec1.Values)
	for i, grad := range v.Jacobian {
		grad.Sub(vec1.Jacobian[i])
	}
}

// Mul performs component-wise multiplication.
func (v *Vector) Mul(v1 anyvec.Vector) {
	vec1 := v.convertVec(v1)
	for i, grad := range v.Jacobian {
		// Product rule.
		vals := v.Values.Copy()
		vals.Mul(vec1.Jacobian[i])
		grad.Mul(vec1.Values)
		grad.Add(vals)
	}
	v.Values.Mul(vec1.Values)
}

// Div performs component-wise division.
func (v *Vector) Div(v1 anyvec.Vector) {
	vec1 := v.convertVec(v1)
	for i, grad := range v.Jacobian {
		// Quotient rule.
		grad.Div(vec1.Values)
		quotPart := v.Values.Copy()
		quotPart.Mul(vec1.Jacobian[i])
		vec1Squared := vec1.Values.Copy()
		anyvec.Pow(vec1Squared, vec1Squared.Creator().MakeNumeric(2))
		quotPart.Div(vec1Squared)
		grad.Sub(quotPart)
	}
	v.Values.Div(vec1.Values)
}

// addNumerics adds two numerics from the underlying
// value creator.
func (v *Vector) addNumerics(n1, n2 anyvec.Numeric) anyvec.Numeric {
	ops := v.CreatorPtr.ValueCreator.NumOps()
	return ops.Add(n1, n2)
}

// clearJacobian sets the jacobian to 0.
func (v *Vector) clearJacobian() {
	zero := v.Values.Creator().MakeVector(v.Values.Len())
	for _, x := range v.Jacobian {
		x.Set(zero)
	}
}

func (v *Vector) convertVec(vec anyvec.Vector) *Vector {
	v1 := vec.(*Vector)
	if len(v.Jacobian) != len(v1.Jacobian) {
		panic(badJacobianErr)
	}
	return v1
}

func (v *Vector) convertNum(num anyvec.Numeric) Numeric {
	n1 := num.(Numeric)
	if len(v.Jacobian) != len(n1.Grad) {
		panic(badJacobianErr)
	}
	return n1
}
