package anyfwd

import "github.com/unixpickle/anyvec"

// Sum sums the vector entries.
func (v *Vector) Sum() anyvec.Numeric {
	res := Numeric{Value: anyvec.Sum(v.Values)}
	for _, grad := range v.Jacobian {
		res.Grad = append(res.Grad, anyvec.Sum(grad))
	}
	return res
}

// Max computes the maximum entry.
func (v *Vector) Max() anyvec.Numeric {
	if v.Len() == 0 {
		return v.Creator().MakeNumeric(0)
	}

	out := v.Creator().MakeVector(1).(*Vector)

	maxMapper := anyvec.MapMax(v.Values, v.Len())
	maxMapper.Map(v.Values, out.Values)

	for i, grad := range v.Jacobian {
		maxMapper.Map(grad, out.Jacobian[i])
	}

	return out.Sum()
}

// AbsSum sums the absolute values of the components.
func (v *Vector) AbsSum() anyvec.Numeric {
	return v.abs().Sum()
}

// AbsMax computes the greatest absolute value.
func (v *Vector) AbsMax() anyvec.Numeric {
	return v.abs().Max()
}

// Norm computes the Euclidean norm.
//
// The derivatives are undefined when the norm is 0.
func (v *Vector) Norm() anyvec.Numeric {
	norm := anyvec.Norm(v.Values)
	res := Numeric{Value: norm}

	invNorm := v.numReciprocal(norm)
	for _, grad := range v.Jacobian {
		scaledGrad := grad.Copy()
		scaledGrad.Scale(invNorm)
		res.Grad = append(res.Grad, scaledGrad.Dot(v.Values))
	}

	return res
}

// MaxIndex returns the index of the maximum element.
func (v *Vector) MaxIndex() int {
	return anyvec.MaxIndex(v.Values)
}

func (v *Vector) abs() *Vector {
	// TODO: if complex vectors are ever available, we will
	// have to do something fancier here.
	// Likely, we should just make Abs() a real API.

	// Create a vector which is -1 for negative values
	// and 1 for positive values.
	signChanger := v.Values.Copy()
	c := signChanger.Creator()
	anyvec.GreaterThan(signChanger, c.MakeNumeric(0))
	signChanger.Scale(c.MakeNumeric(2))
	signChanger.AddScalar(c.MakeNumeric(-1))

	newVec := v.Copy().(*Vector)
	newVec.Values.Mul(signChanger)
	for _, grad := range newVec.Jacobian {
		grad.Mul(signChanger)
	}

	return newVec
}

// numReciprocal computes the reciprocal of a numeric from
// the value creator.
func (v *Vector) numReciprocal(n anyvec.Numeric) anyvec.Numeric {
	recip := v.Values.Creator().MakeVector(1)
	recip.AddScalar(n)
	anyvec.Pow(recip, recip.Creator().MakeNumeric(-1))
	return anyvec.Sum(recip)
}
