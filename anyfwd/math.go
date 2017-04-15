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

func (v *Vector) mulJacobian(rowScaler anyvec.Vector) {
	for _, grad := range v.Jacobian {
		grad.Mul(rowScaler)
	}
}
