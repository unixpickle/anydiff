package anyfwd

import "github.com/unixpickle/anyvec"

// Tanh computes the component-wise hyperbolic tangent.
func (v *Vector) Tanh() {
	anyvec.Tanh(v.Values)
	deriv := v.Values.Copy()
	anyvec.Pow(deriv, deriv.Creator().MakeNumeric(2))
	anyvec.Complement(deriv)
	v.mulJacobian(deriv)
}

// Sin computes the component-wise sine.
func (v *Vector) Sin() {
	deriv := v.Values.Copy()
	anyvec.Cos(deriv)
	anyvec.Sin(v.Values)
	v.mulJacobian(deriv)
}

// Cos computes the component-wise cosine.
func (v *Vector) Cos() {
	deriv := v.Values.Copy()
	anyvec.Sin(deriv)
	deriv.Scale(deriv.Creator().MakeNumeric(-1))
	anyvec.Cos(v.Values)
	v.mulJacobian(deriv)
}

// Exp exponentiates the vector components.
func (v *Vector) Exp() {
	anyvec.Exp(v.Values)
	v.mulJacobian(v.Values)
}

// Log takes the component-wise natural log.
func (v *Vector) Log() {
	deriv := v.Values.Copy()
	anyvec.Pow(deriv, deriv.Creator().MakeNumeric(-1))
	anyvec.Log(v.Values)
	v.mulJacobian(deriv)
}

// Sigmoid takes the component-wise logistic sigmoid.
func (v *Vector) Sigmoid() {
	anyvec.Sigmoid(v.Values)
	deriv := v.Values.Copy()
	anyvec.Complement(deriv)
	deriv.Mul(v.Values)
	v.mulJacobian(deriv)
}

// ClipPos clips the components to non-negative values.
func (v *Vector) ClipPos() {
	mask := v.Values.Copy()
	anyvec.GreaterThan(mask, mask.Creator().MakeNumeric(0))
	v.Values.Mul(mask)
	v.mulJacobian(mask)
}

// Pow raises each component to power p.
//
// Currently, this only supports constant exponents.
func (v *Vector) Pow(p anyvec.Numeric) {
	num := v.convertNum(p)

	if !v.CreatorPtr.constant(num) {
		panic("exponent is not constant")
	}

	pMinusOne := v.addNumerics(num.Value, v.CreatorPtr.ValueCreator.MakeNumeric(-1))

	deriv := v.Values.Copy()
	anyvec.Pow(deriv, pMinusOne)
	deriv.Scale(num.Value)

	v.mulJacobian(deriv)
	anyvec.Pow(v.Values, num.Value)
}

// ElemMax sets each element of v to the max of that
// element and the corresponding element of v1.
func (v *Vector) ElemMax(v1 anyvec.Vector) {
	vec := v.convertVec(v1)

	columnMatrix := func(v1, v2 anyvec.Vector) anyvec.Vector {
		joined := v1.Creator().Concat(v1, v2)
		transposed := joined.Creator().MakeVector(joined.Len())
		anyvec.Transpose(joined, transposed, 2)
		return transposed
	}

	valColumns := columnMatrix(v.Values, vec.Values)
	maxMap := anyvec.MapMax(valColumns, 2)
	maxMap.Map(valColumns, v.Values)

	for i, grad := range v.Jacobian {
		columns := columnMatrix(grad, vec.Jacobian[i])
		maxMap.Map(columns, grad)
	}
}

func (v *Vector) mulJacobian(rowScaler anyvec.Vector) {
	for _, grad := range v.Jacobian {
		grad.Mul(rowScaler)
	}
}
