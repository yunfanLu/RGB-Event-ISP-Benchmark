from ev_rgb_isp.visualize.demosaic_hybridevs_visualization import DemosaicHybridevsBatchVisualization


def get_visulization(config):
    if config.NAME == "demosaic-vis":
        return DemosaicHybridevsBatchVisualization(config)
    else:
        raise NotImplementedError(f"Visualization {config.NAME} is not implemented.")
