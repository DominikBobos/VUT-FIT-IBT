"""     
Implements matplotlib scale warped using probit function (e.g. usefull for Detecton Error Tradeoff (DET) plots)
Change scale of a current plot using:

> xscale('probit')
> yscale('probit')

Additional keyword parameter:

> xscale('probit', nbins=10, unit_scale=100, fmt='%s')

  *nbins*
     Prefered number of ticks on the axis

  *unit_scale*
     Divide vaues by unit_scale before transforming then by probit function.
     E.g. use unit_scale=100 if the ploted vaues are in percents
            (i.e. renge (0, 100) rather then probability range (0, 1))

  *fmt*
     String for formating tick labels (default '%s')
"""

__all__ = [
    'ProbitLocator',
    'ProbitScale',
]

#from utils import probit,invprobit,logit
import numpy as np
from matplotlib.ticker import  Locator, NullLocator, NullFormatter, FormatStrFormatter, MaxNLocator, ScalarFormatter
from matplotlib.scale  import  ScaleBase, register_scale
from matplotlib.transforms import Transform

def logit(a):
    return np.log(a / (1.0 - a))


def probit(a):
    from scipy.special import erfinv
    return np.sqrt(2.0) * erfinv(2.0 * a - 1.0)


def invprobit(a):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(a / np.sqrt(2.0)))


class ProbitTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def __init__(self, unit_scale):
        """
        *unit_scale*
            Divide vaues by unit_scale before transforming then by probit function.
            E.g. use unit_scale=100 if the transformed vaues are in percents
            (i.e. renge (0, 100) rather then probability range (0, 1))
        """
        Transform.__init__(self)
        self._unit_scale = unit_scale

    def transform_non_affine(self, a):
        return probit(a / self._unit_scale)

    #def transform(self, a):
    #    return probit(np.array(a) / self._unit_scale)

    def inverted(self):
        return InvertedProbitTransform(self._unit_scale)

class InvertedProbitTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def __init__(self, unit_scale):

        Transform.__init__(self)
        self._unit_scale = unit_scale

    #def transform(self, a):
    #    return invprobit(a) * self._unit_scale

    def transform_non_affine(self, a):
        return invprobit(a) * self._unit_scale

    def inverted(self):
        return ProbitTransform(self._unit_scale)


class ProbitLocator(Locator):
    """
    Locator for scales warped using probit function. Usefull for Detecton Error Tradeoff (DET) plots
    """
    def __init__(self, nbins, unit_scale):
        """
        Keyword args:
        *nbins*
            Prefered number of ticks on the axis

        *unit_scale*
            Divide vaues by unit_scale before transforming then by probit function.
            E.g. use unit_scale=100 if the ploted vaues are in percents
            (i.e. renge (0, 100) rather then probability range (0, 1))
        """
        self.nbins = nbins
        self.unit_scale = unit_scale
        self.backoff_locator = MaxNLocator(nbins=nbins)

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()

        # place ticks liearly in the probit scale
        locs = invprobit(np.linspace(probit(vmin / self.unit_scale),
                                     probit(vmax / self.unit_scale), self.nbins))

        # snap tick locations to "nice values"

        # Old version. Not as nice
        #locs = np.array([((1-float('%.1g' % ((1-n)*2))/2) if n > 0.5 else float('%.1g' % (n*2))/2)
        #                for n in locs], dtype=float) * self.unit_scale

        bt05 = locs > 0.5
        locs[bt05] = 1 - locs[bt05]
        magnitudes = 10**np.floor(np.log10(locs))
        locs =  np.rint(locs / magnitudes * 2) * magnitudes / 2
        locs[bt05] = 1 - locs[bt05]
        locs *= self.unit_scale

        # filter out duplicated locations snapped to the same value
        locs = np.unique(locs)

        # When zoomed to a linear part of probit scale, we can end up with too
        # few ticks. In such case back off to standard linear locator
        if len(locs) < self.nbins - 2:
            self.backoff_locator.set_axis(self.axis)
            return self.raise_if_exceeds(self.backoff_locator())
        return self.raise_if_exceeds(locs)

class ProbitScale(ScaleBase):
    """
    Scale warped using probit function. Usefull for Detecton Error Tradeoff (DET) plots
    """

    name = 'probit'

    def __init__(self, axis, **kwargs):
        """
        Keyword args:
        *nbins*
            Prefered number of ticks on the axis

        *unit_scale*
            Divide vaues by unit_scale before transforming then by probit function.
            E.g. use unit_scale=100 if the ploted vaues are in percents
            (i.e. renge (0, 100) rather then probability range (0, 1))

        *fmt*
            string for formating tick labels (default %s)

        """
        self.axis = axis
        self.nbins = kwargs.pop('nbins', 10)
        self.unit_scale = kwargs.pop('unit_scale', 1.0)
        self.fmt = kwargs.pop('fmt', '%g')

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to reasonable defaults for
        linear scaling.
        """
        axis.set_major_locator(ProbitLocator(nbins=self.nbins, unit_scale=self.unit_scale))
        axis.set_major_formatter(FormatStrFormatter(self.fmt))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        return ProbitTransform(unit_scale=self.unit_scale)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to values between 0 and unit_scale.
        """
        # 0 if the scale corresponts to x axis; 1 otherwise
        axis_num = int(self.axis.axes.yaxis == self.axis)

        # to set v max, find in the plot data the largest float smaller than unit_scale
        max_valid = None
        for l in self.axis.axes.lines:
          data_xy = np.c_[l.get_data()]
          in_range = np.all((data_xy < self.unit_scale) & (data_xy > 0.0), axis=1)
          if np.any(in_range):
              max_valid_tmp = data_xy[in_range, axis_num].max()
              if max_valid is None or max_valid_tmp > max_valid:
                  max_valid = max_valid_tmp

        # No valid maximum value found in plot data (e.g. no line was plotted yet).
        # Set vmax and max_valid to a reasonably large value
        if max_valid is None:
            vmax = max_valid = 0.999 * self.unit_scale

        if not np.isfinite(minpos):
            minpos = 1e-300    # This value should rarely if ever end up with a visible effect.

        if vmin <= 0.0:             vmin = minpos
        if vmin >= self.unit_scale: vmin = max_valid
        if vmax <= 0.0:             vmax = minpos
        if vmax >= self.unit_scale: vmax = max_valid
        return vmin, vmax

register_scale(ProbitScale)

if(__name__=="__main__"):
    pass
